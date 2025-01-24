#define NOMINMAX
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <chrono>
#include <random>
#include <windows.h>

constexpr int ROWS = 6;
constexpr int COLS = 7;
constexpr int MCTS_SIMS = 1000;
constexpr int MAX_DEPTH = 8;
constexpr float C_PUCT = 2.2f;

HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

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
    
    bool check_pattern(int r, int c, int dr, int dc, char player, int length) const {
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr * i;
            int nc = c + dc * i;
            if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                if (board[nr][nc] == player) count++;
                else if (board[nr][nc] != ' ') return false;
            }
        }
        return count >= length;
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
                return check_win(r, col);
            }
        }
        return false;
    }

    bool check_win(int r, int c) const {
        const int dirs[4][2] = {{0,1}, {1,0}, {1,1}, {-1,1}};
        for (auto [dr, dc] : dirs) {
            if (check_pattern(r, c, dr, dc, board[r][c], 4)) return true;
        }
        return false;
    }

    bool would_win(int col, char player) const {
        GameState temp = *this;
        return temp.make_move(col, player);
    }

    int detect_threats(char opponent, int required) const {
        const int dirs[4][2] = {{0,1}, {1,0}, {1,1}, {-1,1}};
        
        for (int c = 0; c < COLS; ++c) {
            if (board[0][c] != ' ') continue;
            
            // Find landing row
            int r = ROWS-1;
            while (r >= 0 && board[r][c] != ' ') r--;
            
            // Check if this move creates a threat
            for (auto [dr, dc] : dirs) {
                if (check_pattern(r, c, dr, dc, opponent, required) ||
                    check_pattern(r, c, -dr, -dc, opponent, required)) {
                    return c;
                }
            }
        }
        return -1;
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
    std::mt19937 gen;
    
    float ucb_score(const MCTSNode* node) const {
        if (node->visit_count == 0) return INFINITY;
        return (node->value_sum / node->visit_count) + 
               C_PUCT * node->prior * std::sqrt(node->parent->visit_count) / (1 + node->visit_count);
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
            
            for (int move : moves) {
                if (sim_state.would_win(move, current)) 
                    return current == 'Y' ? 1.0f : -1.0f;
            }
            
            int threat = sim_state.detect_threats(current == 'Y' ? 'R' : 'Y', 3);
            if (threat != -1 && std::find(moves.begin(), moves.end(), threat) != moves.end()) {
                sim_state.make_move(threat, current);
                current = current == 'Y' ? 'R' : 'Y';
                continue;
            }

            threat = sim_state.detect_threats(current == 'Y' ? 'R' : 'Y', 2);
            if (threat != -1 && std::find(moves.begin(), moves.end(), threat) != moves.end()) {
                sim_state.make_move(threat, current);
                current = current == 'Y' ? 'R' : 'Y';
                continue;
            }

            int move = moves[dist(gen) % moves.size()];
            sim_state.make_move(move, current);
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
    MCTS() : gen(std::random_device{}()) {}
    
    int search(const GameState& game_state) {
        state = game_state;
        root = std::make_unique<MCTSNode>();
        expand(root.get());
        
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

int minimax(GameState& state, int depth, int alpha, int beta, bool maximizing) {
    auto moves = state.get_valid_moves();
    if (moves.empty() || depth == 0) return 0;

    int best_value = maximizing ? -100000 : 100000;
    char player = maximizing ? 'Y' : 'R';

    int threat = state.detect_threats(maximizing ? 'R' : 'Y', 3);
    if (threat != -1 && depth >= 2) {
        return maximizing ? -95000 : 95000;
    }

    threat = state.detect_threats(maximizing ? 'R' : 'Y', 2);
    if (threat != -1 && depth >= 4) {
        return maximizing ? -90000 : 90000;
    }

    for (int move : moves) {
        bool win = state.make_move(move, player);
        if (win) {
            state.undo_move(move);
            return maximizing ? 100000 : -100000;
        }
        
        int value = minimax(state, depth-1, alpha, beta, !maximizing);
        state.undo_move(move);

        if (maximizing) {
            best_value = std::max(best_value, value);
            alpha = std::max(alpha, value);
        } else {
            best_value = std::min(best_value, value);
            beta = std::min(beta, value);
        }
        
        if (beta <= alpha) break;
    }
    return best_value;
}

int hybrid_decision(const GameState& state) {
    auto moves = state.get_valid_moves();
    if (moves.empty()) return -1;

    for (int move : moves) {
        if (state.would_win(move, 'Y')) return move;
    }

    for (int move : moves) {
        if (state.would_win(move, 'R')) return move;
    }

    int threat = state.detect_threats('R', 3);
    if (threat != -1) return threat;

    threat = state.detect_threats('R', 2);
    if (threat != -1) return threat;

    MCTS mcts;
    GameState mm_state = state;
    
    int best_mm = -100000;
    int mm_move = moves[0];
    
    for (int move : moves) {
        mm_state.make_move(move, 'Y');
        int score = minimax(mm_state, MAX_DEPTH, -100000, 100000, false);
        mm_state.undo_move(move);
        
        if (score > best_mm) {
            best_mm = score;
            mm_move = move;
        }
    }

    return (best_mm >= 50000) ? mm_move : mcts.search(state);
}

void display_board(const GameState& state) {
    const auto& board = state.get_board();
    system("cls");
    
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            std::cout << "| ";
            char piece = board[r][c];
            if (piece != ' ') {
                SetConsoleTextAttribute(hConsole, piece == 'R' ? 12 : 14);
                std::cout << piece;
                SetConsoleTextAttribute(hConsole, 7);
            } else {
                std::cout << ' ';
            }
            std::cout << ' ';
        }
        std::cout << "|\n";
    }
    
    std::cout << "+---+---+---+---+---+---+---+\n";
    std::cout << "  1   2   3   4   5   6   7\n";
}

int main() {
    GameState state;
    char current = 'R';

    while (true) {
        display_board(state);
        auto moves = state.get_valid_moves();
        if (moves.empty()) {
            std::cout << "Game Over! It's a draw!\n";
            break;
        }

        if (current == 'R') {
            int move;
            do {
                std::cout << "Your move (1-7): ";
                std::cin >> move;
                move--;
            } while (move < 0 || move >= COLS || 
                     std::find(moves.begin(), moves.end(), move) == moves.end());
            
            if (state.make_move(move, 'R')) {
                display_board(state);
                std::cout << "Congratulations! You win!\n";
                system("pause");
                return 0;
            }
        } else {
            auto start = std::chrono::high_resolution_clock::now();
            int move = hybrid_decision(state);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (state.make_move(move, 'Y')) {
                display_board(state);
                std::cout << "AI wins!\n";
                system("pause");
                return 0;
            }
            
            std::cout << "AI chose column " << (move+1) 
                      << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
                      << "ms)\n";
        }

        current = (current == 'R') ? 'Y' : 'R';
    }
    system("pause");
    return 0;
}