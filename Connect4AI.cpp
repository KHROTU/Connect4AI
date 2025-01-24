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
constexpr int MCTS_SIMS = 1200;
constexpr int MAX_DEPTH = 9;
constexpr float C_PUCT = 2.4f;

HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
const WORD PLAYER_COLOR = 12;
const WORD AI_COLOR = 14;
const WORD BOARD_COLOR = 7;

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
    
    bool check_line(int r, int c, int dr, int dc, char player, int length) const {
        int count = 0;
        for (int i = -3; i <= 3; ++i) {
            int nr = r + dr*i;
            int nc = c + dc*i;
            if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                if (board[nr][nc] == player) {
                    if (++count >= length) return true;
                } else if (board[nr][nc] != ' ') {
                    count = 0;
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
                return check_win(r, col);
            }
        }
        return false;
    }

    bool check_win(int r, int c) const {
        const int dirs[4][2] = {{0,1}, {1,0}, {1,1}, {-1,1}};
        for (auto [dr, dc] : dirs) {
            if (check_line(r, c, dr, dc, board[r][c], 4)) return true;
        }
        return false;
    }

    bool would_win(int col, char player) const {
        GameState temp = *this;
        return temp.make_move(col, player);
    }

    int threat_scan(char opponent) const {
        const int dirs[4][2] = {{0,1}, {1,0}, {1,1}, {-1,1}};
        
        for (int c = 0; c < COLS; ++c) {
            if (board[0][c] != ' ') continue;
            
            int r = ROWS-1;
            while (r >= 0 && board[r][c] != ' ') r--;
            
            for (auto [dr, dc] : dirs) {
                if (check_line(r, c, dr, dc, opponent, 3) ||
                    check_line(r, c, -dr, -dc, opponent, 3))
                    return c;
            }
        }
        return -1;
    }

    int evaluate_position(char player) const {
        int score = 0;
        const int dirs[4][2] = {{0,1}, {1,0}, {1,1}, {-1,1}};
        
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (board[r][c] == player) {
                    for (auto [dr, dc] : dirs) {
                        int sequence = 1;
                        bool open_start = true, open_end = true;
                        
                        for (int i = 1; i < 4; ++i) {
                            int nr = r - dr*i;
                            int nc = c - dc*i;
                            if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) open_start = false;
                        }
                        
                        for (int i = 1; i < 4; ++i) {
                            int nr = r + dr*i;
                            int nc = c + dc*i;
                            if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                                if (board[nr][nc] == player) sequence++;
                                else if (board[nr][nc] != ' ') open_end = false;
                            } else {
                                open_end = false;
                            }
                        }
                        
                        if (sequence >= 2) {
                            int weight = open_start + open_end;
                            score += (sequence == 2) ? weight * 10 : 
                                    (sequence == 3) ? weight * 100 : 1000;
                        }
                    }
                }
            }
        }
        return score;
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
        
        for (int i = 0; i < 15; ++i) {
            auto moves = sim_state.get_valid_moves();
            if (moves.empty()) return 0.0f;
            
            for (int move : moves) {
                if (sim_state.would_win(move, current)) 
                    return current == 'Y' ? 1.0f : -1.0f;
            }
            
            int threat = sim_state.threat_scan(current == 'Y' ? 'R' : 'Y');
            if (threat != -1) {
                sim_state.make_move(threat, current);
                current = current == 'Y' ? 'R' : 'Y';
                continue;
            }

            int best_move = -1;
            int best_score = -1;
            for (int move : moves) {
                int score = sim_state.evaluate_position(current);
                if (score > best_score) {
                    best_score = score;
                    best_move = move;
                }
            }
            
            if (best_move != -1) {
                sim_state.make_move(best_move, current);
                current = current == 'Y' ? 'R' : 'Y';
            } else {
                break;
            }
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
    if (moves.empty() || depth == 0) 
        return state.evaluate_position('Y') - state.evaluate_position('R');

    if (maximizing) {
        int value = -1000000;
        for (int move : moves) {
            GameState temp = state;
            temp.make_move(move, 'Y');
            value = std::max(value, minimax(temp, depth-1, alpha, beta, false));
            alpha = std::max(alpha, value);
            if (beta <= alpha) break;
        }
        return value;
    } else {
        int value = 1000000;
        for (int move : moves) {
            GameState temp = state;
            temp.make_move(move, 'R');
            value = std::min(value, minimax(temp, depth-1, alpha, beta, true));
            beta = std::min(beta, value);
            if (beta <= alpha) break;
        }
        return value;
    }
}

int hybrid_decision(const GameState& state) {
    auto moves = state.get_valid_moves();
    if (moves.empty()) return -1;

    // Immediate win check using const method
    for (int move : moves) {
        if (state.would_win(move, 'Y')) {
            return move;
        }
    }

    // Block opponent win using const method
    for (int move : moves) {
        if (state.would_win(move, 'R')) {
            return move;
        }
    }

    // Block threats using const method
    int threat = state.threat_scan('R');
    if (threat != -1) return threat;

    MCTS mcts;
    GameState mm_state = state;
    
    int best_score = -1000000;
    int best_move = moves[0];
    
    for (int move : moves) {
        GameState temp = mm_state;
        temp.make_move(move, 'Y');
        int score = minimax(temp, MAX_DEPTH, -1000000, 1000000, false);
        
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }

    return (best_score >= 5000) ? best_move : mcts.search(state);
}

void display_board(const GameState& state) {
    const auto& board = state.get_board();
    system("cls");
    
    SetConsoleTextAttribute(hConsole, BOARD_COLOR);
    std::cout << "\n  ";
    for (int c = 0; c < COLS; ++c)
        std::cout << "----";
    std::cout << "-\n";
    
    for (int r = 0; r < ROWS; ++r) {
        std::cout << "  |";
        for (int c = 0; c < COLS; ++c) {
            char piece = board[r][c];
            if (piece == 'R') SetConsoleTextAttribute(hConsole, PLAYER_COLOR);
            else if (piece == 'Y') SetConsoleTextAttribute(hConsole, AI_COLOR);
            else SetConsoleTextAttribute(hConsole, BOARD_COLOR);
            
            std::cout << " " << piece << " ";
            SetConsoleTextAttribute(hConsole, BOARD_COLOR);
            std::cout << "|";
        }
        std::cout << "\n  ";
        for (int c = 0; c < COLS; ++c)
            std::cout << "----";
        std::cout << "-\n";
    }
    
    SetConsoleTextAttribute(hConsole, BOARD_COLOR);
    std::cout << "   ";
    for (int c = 0; c < COLS; ++c)
        std::cout << " " << c+1 << "  ";
    std::cout << "\n\n";
}

int main() {
    SetConsoleTextAttribute(hConsole, BOARD_COLOR);
    std::cout << "CONNECT 4 - AI EDITION\n";
    std::cout << "1. Player First\n2. AI First\nChoice: ";
    
    int choice;
    std::cin >> choice;
    char current = (choice == 1) ? 'R' : 'Y';
    
    GameState state;
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
            } while (std::find(moves.begin(), moves.end(), move) == moves.end());
            
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