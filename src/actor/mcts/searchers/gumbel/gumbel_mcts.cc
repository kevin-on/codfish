#include "actor/mcts/searchers/gumbel/gumbel_mcts.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "chess/position.h"
#include "lc0/move_index.h"

namespace engine {
namespace {

std::vector<float> GenerateGumbelSamples(size_t num_samples) {
  constexpr float kEps = std::numeric_limits<float>::epsilon();
  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> uniform(kEps, 1.0f - kEps);
  std::vector<float> samples(num_samples);
  std::generate(samples.begin(), samples.end(), [&]() {
    const float u = uniform(rng);
    return -std::log(-std::log(u));
  });
  return samples;
}

float GameResultToValue(const lczero::Position& position,
                        lczero::GameResult result) {
  switch (result) {
    case lczero::GameResult::UNDECIDED:
    case lczero::GameResult::DRAW:
      return 0.0f;
    case lczero::GameResult::WHITE_WON:
      return position.IsBlackToMove() ? -1.0f : 1.0f;
    case lczero::GameResult::BLACK_WON:
      return position.IsBlackToMove() ? 1.0f : -1.0f;
  }

  assert(false);
  return 0.0f;
}

std::vector<float> Softmax(const std::vector<float>& logits) {
  if (logits.empty()) {
    return {};
  }

  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probs(logits.size());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }

  const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (float& prob : probs) {
    prob *= inv_sum;
  }
  return probs;
}

}  // namespace

class GumbelMCTS::Impl {
  struct Node;

 public:
  explicit Impl(GumbelMCTSConfig config) : config_(config) {
    assert(config_.num_action > 0);
    assert(config_.num_simulation > 0);
    root_history_.Reset(
        lczero::Position::FromFen(lczero::ChessBoard::kStartposFen));
    root_ = CreateRootNode(root_history_.Last(), root_history_);
  }

  SearchCoroutine Run() {
    if (root_->IsTerminal()) {
      co_return BuildTerminalSearchResult(root_history_);
    }

    if (root_->NeedsEvaluation()) {
      EvalRequest eval_request;
      eval_request.items.emplace_back();
      FillEvalItem(eval_request.items.back(), root_history_);
      EvalResponse eval_response = co_yield std::move(eval_request);
      assert(eval_response.items.size() == 1);
      ExpandNode(root_.get(), eval_response.items[0].policy_logits,
                 eval_response.items[0].wdl_probs);
    }

    assert(!root_->edges.empty());

    std::vector<float> gumbel_samples =
        GenerateGumbelSamples(root_->edges.size());
    std::vector<size_t> candidate_actions =
        SelectInitialRootActions(*root_, gumbel_samples);
    assert(candidate_actions.size() > 0);
    if (candidate_actions.size() == 1) {
      co_return BuildSearchResult(candidate_actions[0]);
    }

    const size_t num_phases =
        static_cast<size_t>(std::ceil(std::log2(candidate_actions.size())));
    assert(num_phases > 0);

    for (size_t phase = 0; phase < num_phases; ++phase) {
      // NOTE: This is a simplified Sequential Halving budget split. Unlike the
      // paper's budget accounting, we do not track remaining simulations across
      // phases here, so total root visits can exceed config_.num_simulation.
      const size_t visits_per_action = std::max<size_t>(
          1, config_.num_simulation / (num_phases * candidate_actions.size()));

      for (size_t visit = 0; visit < visits_per_action; ++visit) {
        EvalRequest eval_request;
        eval_request.items.reserve(candidate_actions.size());
        std::vector<Node*> eval_leaf_nodes;
        eval_leaf_nodes.reserve(candidate_actions.size());

        for (size_t action : candidate_actions) {
          lczero::PositionHistory history(root_history_);
          history.Append(root_->edges[action].move);

          Node* node = EnsureChild(root_.get(), action, history);
          while (node->IsExpanded()) {
            const size_t next_action = SelectNonRootAction(node);
            history.Append(node->edges[next_action].move);
            node = EnsureChild(node, next_action, history);
          }

          if (node->IsTerminal()) {
            Backpropagate(node, node->terminal_value);
            continue;
          }

          assert(node->NeedsEvaluation());
          eval_request.items.emplace_back();
          FillEvalItem(eval_request.items.back(), history);
          eval_leaf_nodes.push_back(node);
        }

        if (eval_leaf_nodes.empty()) {
          continue;
        }

        EvalResponse eval_response = co_yield std::move(eval_request);
        assert(eval_response.items.size() == eval_leaf_nodes.size());

        for (size_t i = 0; i < eval_leaf_nodes.size(); ++i) {
          assert(eval_leaf_nodes[i]->NeedsEvaluation());
          ExpandNode(eval_leaf_nodes[i], eval_response.items[i].policy_logits,
                     eval_response.items[i].wdl_probs);
          Backpropagate(eval_leaf_nodes[i], eval_leaf_nodes[i]->GetWdlValue());
        }
      }

      // Sequential Halving
      PruneRootActionsSequentialHalving(candidate_actions, gumbel_samples);
      if (candidate_actions.size() == 1) {
        break;
      }
    }

    assert(candidate_actions.size() == 1);
    co_return BuildSearchResult(candidate_actions[0]);
  }

  void CommitMove(lczero::Move move) {
    assert(root_ != nullptr);

    if (root_->IsExpanded()) {
      for (size_t i = 0; i < root_->edges.size(); ++i) {
        if (root_->edges[i].move != move) {
          continue;
        }
        lczero::PositionHistory history(root_history_);
        history.Append(root_->edges[i].move);
        EnsureChild(root_.get(), i, history);
        root_ = std::move(root_->edges[i].child);
        root_->parent = nullptr;
        root_history_ = std::move(history);
        root_->pos = root_history_.Last();
        return;
      }
    }

    root_history_.Append(move);
    root_ = CreateRootNode(root_history_.Last(), root_history_);
  }

 private:
  enum class NodeState { kNew, kExpanded, kTerminal };

  struct Edge {
    lczero::Move move;
    float prior = 0.0f;
    std::unique_ptr<Node> child;
  };

  struct Node {
    explicit Node(const lczero::Position& position, Node* parent_node = nullptr)
        : pos(position), parent(parent_node) {}

    float GetQ() const {
      return visit_cnt > 0 ? value_sum / static_cast<float>(visit_cnt) : 0.0f;
    }

    float GetWdlValue() const { return wdl_probs[0] - wdl_probs[2]; }

    bool NeedsEvaluation() const { return state == NodeState::kNew; }
    bool IsExpanded() const { return state == NodeState::kExpanded; }
    bool IsTerminal() const { return state == NodeState::kTerminal; }

    lczero::Position pos;
    std::vector<Edge> edges;
    std::vector<float> logits;
    std::array<float, 3> wdl_probs = {0.0f, 0.0f, 0.0f};
    Node* parent = nullptr;
    float value_sum = 0.0f;
    float terminal_value = 0.0f;
    uint32_t visit_cnt = 0;
    NodeState state = NodeState::kNew;
  };

  void FillEvalItem(EvalRequestItem& item,
                    const lczero::PositionHistory& history) const {
    const auto positions = history.GetPositions();
    const size_t num_positions =
        std::min(positions.size(), static_cast<size_t>(kHistoryLength));
    const size_t start = positions.size() - num_positions;
    for (size_t i = 0; i < num_positions; ++i) {
      item.positions[i] = positions[start + i];
    }
    item.len = static_cast<uint8_t>(num_positions);
  }

  std::unique_ptr<Node> CreateRootNode(
      const lczero::Position& position,
      const lczero::PositionHistory& history) const {
    auto node = std::make_unique<Node>(position);
    InitializeNodeState(node.get(), history);
    return node;
  }

  void InitializeNodeState(Node* node,
                           const lczero::PositionHistory& history) const {
    assert(node != nullptr);
    const lczero::GameResult game_result = history.ComputeGameResult();
    if (game_result == lczero::GameResult::UNDECIDED) {
      node->state = NodeState::kNew;
      node->terminal_value = 0.0f;
      return;
    }

    node->state = NodeState::kTerminal;
    node->terminal_value = GameResultToValue(history.Last(), game_result);
  }

  void ExpandNode(Node* node, const std::vector<float>& policy_logits,
                  const std::vector<float>& wdl_probs) {
    assert(node != nullptr);
    assert(policy_logits.size() == lczero::kPolicySize);
    assert(wdl_probs.size() == 3);
    assert(node->NeedsEvaluation());

    auto legal_moves = node->pos.GetBoard().GenerateLegalMoves();
    node->edges.resize(legal_moves.size());
    node->logits.resize(legal_moves.size());
    assert(!node->edges.empty());

    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < legal_moves.size(); ++i) {
      const lczero::Move move = legal_moves[i];
      const uint16_t idx = lczero::MoveToNNIndex(move, 0);
      const float logit = policy_logits[idx];
      node->edges[i].move = move;
      node->logits[i] = logit;
      max_logit = std::max(max_logit, logit);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < node->logits.size(); ++i) {
      node->edges[i].prior = std::exp(node->logits[i] - max_logit);
      sum += node->edges[i].prior;
    }

    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (auto& edge : node->edges) {
      edge.prior *= inv_sum;
    }

    node->wdl_probs = {wdl_probs[0], wdl_probs[1], wdl_probs[2]};
    node->state = NodeState::kExpanded;
  }

  Node* EnsureChild(Node* node, size_t edge_idx,
                    const lczero::PositionHistory& history) const {
    assert(node != nullptr);
    assert(edge_idx < node->edges.size());

    Edge& edge = node->edges[edge_idx];
    if (edge.child == nullptr) {
      edge.child = std::make_unique<Node>(history.Last(), node);
      InitializeNodeState(edge.child.get(), history);
    }
    return edge.child.get();
  }

  void Backpropagate(Node* node, float value) const {
    for (Node* cur = node; cur != nullptr; cur = cur->parent) {
      cur->value_sum += value;
      cur->visit_cnt += 1;
      value = -value;
    }
  }

  size_t SelectNonRootAction(const Node* node) const {
    // NOTE: Non-root action selection intentionally stays AlphaZero-style PUCT
    // for now. This implementation does not yet use the paper's Section 5
    // completed-Q deterministic selection at non-root nodes.

    assert(node != nullptr);
    assert(node->IsExpanded());

    uint32_t total_visits = 0;
    for (const auto& edge : node->edges) {
      total_visits += edge.child != nullptr ? edge.child->visit_cnt : 0;
    }

    auto score = [&](size_t action) {
      const Edge& edge = node->edges[action];
      const uint32_t child_visits =
          edge.child != nullptr ? edge.child->visit_cnt : 0;
      const float q_value = edge.child != nullptr ? -edge.child->GetQ() : 0.0f;
      const float exploration =
          config_.c_puct * edge.prior *
          std::sqrt(1.0f + static_cast<float>(total_visits)) /
          (1.0f + static_cast<float>(child_visits));
      return q_value + exploration;
    };

    size_t best_action = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (size_t action = 0; action < node->edges.size(); ++action) {
      const float action_score = score(action);
      if (action_score > best_score) {
        best_score = action_score;
        best_action = action;
      }
    }

    return best_action;
  }

  SearchResult BuildSearchResult(size_t selected_action) const {
    assert(root_ != nullptr);
    assert(root_->IsExpanded());
    assert(selected_action < root_->edges.size());

    std::vector<float> improved_logits(root_->logits.size());
    const float root_value = root_->GetWdlValue();
    for (size_t action = 0; action < root_->edges.size(); ++action) {
      const Node* child = root_->edges[action].child.get();
      const bool visited = child != nullptr && child->visit_cnt > 0;
      const float completed_q = visited ? -child->GetQ() : root_value;
      improved_logits[action] =
          root_->logits[action] + Sigma(completed_q, root_.get());
    }

    SearchResult result;
    result.root_history = root_history_;
    result.selected_move = root_->edges[selected_action].move;
    result.legal_moves.reserve(root_->edges.size());
    for (const auto& edge : root_->edges) {
      result.legal_moves.push_back(edge.move);
    }
    result.improved_policy = Softmax(improved_logits);
    result.game_result = lczero::GameResult::UNDECIDED;
    return result;
  }

  SearchResult BuildTerminalSearchResult(
      const lczero::PositionHistory& history) const {
    const lczero::GameResult game_result = history.ComputeGameResult();
    assert(game_result != lczero::GameResult::UNDECIDED);

    SearchResult result;
    result.game_result = game_result;
    return result;
  }

  void PruneRootActionsSequentialHalving(
      std::vector<size_t>& candidate_actions,
      const std::vector<float>& gumbel_samples) const {
    std::vector<float> action_scores(root_->edges.size());
    for (size_t action : candidate_actions) {
      const Node* child = root_->edges[action].child.get();
      const float q_value = child != nullptr ? -child->GetQ() : 0.0f;
      action_scores[action] = gumbel_samples[action] + root_->logits[action] +
                              Sigma(q_value, root_.get());
    }

    const size_t num_actions_to_keep = std::max<size_t>(
        1, static_cast<size_t>((candidate_actions.size() + 1) / 2));
    std::partial_sort(candidate_actions.begin(),
                      candidate_actions.begin() + num_actions_to_keep,
                      candidate_actions.end(), [&](size_t lhs, size_t rhs) {
                        return action_scores[lhs] > action_scores[rhs];
                      });
    candidate_actions.resize(num_actions_to_keep);
  }

  float Sigma(float q_value, const Node* node) const {
    // sigma function defined in Eq (8) in the gumbel alphazero paper
    assert(node != nullptr);
    assert(node->IsExpanded());

    uint32_t max_visits = 0;
    for (const auto& edge : node->edges) {
      if (edge.child != nullptr) {
        max_visits = std::max<uint32_t>(max_visits, edge.child->visit_cnt);
      }
    }

    return (config_.c_visit + max_visits) * config_.c_scale * q_value;
  }

  std::vector<size_t> SelectInitialRootActions(
      const Node& root, const std::vector<float>& gumbel_samples) const {
    assert(gumbel_samples.size() == root.logits.size());
    std::vector<size_t> candidate_actions(root.logits.size());
    for (size_t i = 0; i < candidate_actions.size(); ++i) {
      candidate_actions[i] = i;
    }

    const size_t num_actions_to_select =
        std::min(static_cast<size_t>(config_.num_action), root.logits.size());

    std::partial_sort(candidate_actions.begin(),
                      candidate_actions.begin() + num_actions_to_select,
                      candidate_actions.end(), [&](size_t lhs, size_t rhs) {
                        return gumbel_samples[lhs] + root.logits[lhs] >
                               gumbel_samples[rhs] + root.logits[rhs];
                      });
    candidate_actions.resize(num_actions_to_select);
    return candidate_actions;
  }

  GumbelMCTSConfig config_;
  lczero::PositionHistory root_history_;
  std::unique_ptr<Node> root_;
};

GumbelMCTS::GumbelMCTS(GumbelMCTSConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

GumbelMCTS::~GumbelMCTS() = default;

SearchCoroutine GumbelMCTS::Run() { return impl_->Run(); }

void GumbelMCTS::CommitMove(lczero::Move move) { impl_->CommitMove(move); }

}  // namespace engine
