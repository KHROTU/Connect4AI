#define NOMINMAX
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <chrono>
#include <random>

constexpr int ROWS = 6;
constexpr int COLS = 7;
constexpr int MCTS_SIMS = 400;
constexpr int MAX_DEPTH = 5;
constexpr float C_PUCT = 1.5f;

struct MCTSNode {
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> valid_moves;
    int visit_count = 0;
    float value_sum = 0;
    float prior = 0;
    
    explicit MCTSNode(MCTSNode* p = nullptr) : parent(p) {}
};

class GameState {
private:
    std::vector<std::vector<char>> board;
    std::mt19937 gen{std::random_device{}()};

    bool check_win(int row, int col, char player) const {
        const int dirs[4][2] = {{0,1}, {1,0}, {1,1}, {-1,1}};
        for (auto [dr, dc] : dirs) {
            int count = 1;
            for (int d = -1; d <= 1; d += 2) {
                for (int i = 1; ; i++) {
                    int r = row + dr * i * d;
                    int c = col + dc * i * d;
                    if (r < 0 || r >= ROWS || c < 0 || c >= COLS || board[r][c] != player) break;
                    if (++count >= 4) return true;
                }
            }
        }
        return false;
    }

public:
    GameState() : board(ROWS, std::vector<char>(COLS, ' ')) {}

    std::vector<int> get_valid_moves() const {
        std::vector<int> moves;
        for (int c = 0; c < COLS; ++c)
            if (board[0][c] == ' ') moves.push_back(c);
        return moves;
    }

    bool make_move(int col, char player) {
        for (int r = ROWS-1; r >= 0; --r) {
            if (board[r][col] == ' ') {
                board[r][col] = player;
                return check_win(r, col, player);
            }
        }
        return false;
    }

    void undo_move(int col) {
        for (int r = 0; r < ROWS; ++r) {
            if (board[r][col] != ' ') {
                board[r][col] = ' ';
                return;
            }
        }
    }

    const auto& get_board() const { return board; }
};

class MCTS {
private:
    std::unique_ptr<MCTSNode> root;
    GameState state;

    float ucb_score(const MCTSNode* node) const {
        if (node->visit_count == 0) return INFINITY;
        return (node->value_sum / node->visit_count) + 
               C_PUCT * node->prior * std::sqrt(root->visit_count) / (1 + node->visit_count);
    }

    MCTSNode* select(MCTSNode* node) {
        while (!node->children.empty()) {
            auto best = std::max_element(node->children.begin(), node->children.end(),
                [&](const auto& a, const auto& b) { return ucb_score(a.get()) < ucb_score(b.get()); });
            node = best->get();
        }
        return node;
    }

    void expand(MCTSNode* node) {
        auto moves = state.get_valid_moves();
        node->valid_moves = moves;
        for (int move : moves) {
            auto child = std::make_unique<MCTSNode>(node);
            child->prior = 1.0f / moves.size();
            node->children.push_back(std::move(child));
        }
    }

    float simulate() {
        GameState sim_state = state;
        char current = 'Y';
        std::uniform_int_distribution<int> dist(0, COLS-1);
        
        for (int i = 0; i < 12; ++i) {
            auto moves = sim_state.get_valid_moves();
            if (moves.empty()) return 0.0f;
            
            int move = moves[dist(gen) % moves.size()];
            bool win = sim_state.make_move(move, current);
            if (win) return current == 'Y' ? 1.0f : -1.0f;
            current = current == 'Y' ? 'R' : 'Y';
        }
        return 0.0f;
    }

    void backpropagate(MCTSNode* node, float value) {
        while (node) {
            node->visit_count++;
            node->value_sum += value;
            value = -value;
            node = node->parent;
        }
    }

public:
    int search(const GameState& game_state) {
        state = game_state;
        root = std::make_unique<MCTSNode>();
        
        for (int i = 0; i < MCTS_SIMS; ++i) {
            MCTSNode* node = select(root.get());
            
            if (node->visit_count == 0) {
                float value = simulate();
                backpropagate(node, value);
            } else {
                expand(node);
                if (!node->children.empty()) {
                    float value = simulate();
                    backpropagate(node->children[0].get(), value);
                }
            }
        }

        auto best = std::max_element(root->children.begin(), root->children.end(),
            [](const auto& a, const auto& b) { return a->visit_count < b->visit_count; });
        
        return root->valid_moves[std::distance(root->children.begin(), best)];
    }
};

int minimax(GameState& state, int depth, float alpha, float beta, bool maximizing) {
    if (depth == 0) return 0;
    
    auto moves = state.get_valid_moves();
    if (moves.empty()) return 0;

    int best_value = maximizing ? -1000 : 1000;
    char player = maximizing ? 'Y' : 'R';

    for (int move : moves) {
        bool win = state.make_move(move, player);
        if (win) {
            state.undo_move(move);
            return maximizing ? 1000 : -1000;
        }
        
        int value = minimax(state, depth-1, alpha, beta, !maximizing);
        state.undo_move(move);

        if (maximizing) {
            best_value = std::max(best_value, value);
            alpha = std::max(alpha, (float)value);
        } else {
            best_value = std::min(best_value, value);
            beta = std::min(beta, (float)value);
        }
        
        if (beta <= alpha) break;
    }
    return best_value;
}

int hybrid_decision(const GameState& state) {
    // MCTS decision
    MCTS mcts;
    int mcts_move = mcts.search(state);
    
    // Minimax decision
    GameState mm_state = state;
    int best_mm = -1000;
    int mm_move = -1;
    auto moves = mm_state.get_valid_moves();
    
    for (int move : moves) {
        mm_state.make_move(move, 'Y');
        int score = minimax(mm_state, MAX_DEPTH, -1000, 1000, false);
        mm_state.undo_move(move);
        
        if (score > best_mm) {
            best_mm = score;
            mm_move = move;
        }
    }

    // Combine results
    return (best_mm >= 500) ? mm_move : mcts_move;
}

int main() {
    GameState state;
    char human = 'R';
    char ai = 'Y';
    char current = 'R';

    while (true) {
        system("cls");
        const auto& board = state.get_board();
        
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c)
                std::cout << "| " << board[r][c] << " ";
            std::cout << "|\n";
        }
        std::cout << "+---+---+---+---+---+---+---+\n";

        if (current == human) {
            int move;
            do {
                std::cout << "Column (1-7): ";
                std::cin >> move;
                move--;
            } while (move < 0 || move >= COLS || state.get_valid_moves()[move] != move);
            
            if (state.make_move(move, human)) {
                std::cout << "Human wins!\n";
                break;
            }
        } else {
            auto start = std::chrono::high_resolution_clock::now();
            int move = hybrid_decision(state);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::cout << "AI chose: " << (move+1) << " (" 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
                      << "ms)\n";
            
            if (state.make_move(move, ai)) {
                std::cout << "AI wins!\n";
                break;
            }
        }

        if (state.get_valid_moves().empty()) {
            std::cout << "Draw!\n";
            break;
        }

        current = (current == human) ? ai : human;
    }
    return 0;
}