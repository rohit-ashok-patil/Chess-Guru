import os
import time
import chess
import torch
import numpy as np
import random
import chess.engine
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from build_model import MxModel
from create_board_position_tensor import fen_to_maia2_tensor

# --- Flask & Database Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_123')

# Database: Uses Render Postgres if available, else local SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
if app.config['SQLALCHEMY_DATABASE_URI'] and app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    rating = db.Column(db.Integer, default=1200)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Model and Move Index Setup ---
with open("generated_uci_moves.txt") as f:
    uci_moves = [line.strip() for line in f if line.strip()]
index_to_uci = {i: move for i, move in enumerate(uci_moves)}
uci_to_index = {move: i for i, move in enumerate(uci_moves)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (Optimized for 57MB file)
model = MxModel(in_channels=60, n_blocks=16, n_moves=1715, channels=192)
model.load_state_dict(torch.load("models/best_model_1_3.pth", map_location=device, weights_only=True))
model.eval()
model.to(device)

# --- Stockfish Setup (Linux/Render Compatible) ---
STOCKFISH_PATH = os.path.join(os.getcwd(), "stockfish/stockfish-ubuntu-x86-64-avx2")

# Automatic permission fix for Linux
if os.name == 'posix' and os.path.exists(STOCKFISH_PATH):
    import stat
    st = os.stat(STOCKFISH_PATH)
    os.chmod(STOCKFISH_PATH, st.st_mode | stat.S_IEXEC)

def get_stockfish_eval(fen, multipv=1, depth=15):
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            if isinstance(info, dict): info = [info]
            results = []
            for i in info:
                score_info = i["score"].white()
                score = (10000 if score_info.mate() > 0 else -10000) if score_info.is_mate() else score_info.score(mate_score=10000)
                move = i["pv"][0].uci() if "pv" in i and i["pv"] else None
                results.append({"move": move, "score": score})
            return results
    except Exception as e:
        return [{"move": None, "score": 0}]

def model_policy_top_moves(board, top_n=3):
    try:
        tensor = fen_to_maia2_tensor(board.fen())
        x = torch.tensor(np.transpose(tensor, (2, 0, 1))[None, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            policy_output = model(x)
            if isinstance(policy_output, tuple): policy_output = policy_output[0]
            probs = torch.softmax(policy_output, dim=1).cpu().numpy().flatten()
        
        legal_uci = [move.uci() for move in board.legal_moves]
        legal_indices = [uci_to_index[m] for m in legal_uci if m in uci_to_index]
        if not legal_indices: return [], []
        
        legal_probs = [(idx, probs[idx]) for idx in legal_indices]
        legal_probs.sort(key=lambda x: x[1], reverse=True)
        top_moves = legal_probs[:min(top_n, len(legal_probs))]
        return [index_to_uci[idx] for idx, _ in top_moves], [float(prob) for _, prob in top_moves]
    except:
        return [], []

def uci_to_move_obj(uci):
    move_obj = {"from": uci[:2], "to": uci[2:4]}
    if len(uci) == 5: move_obj["promotion"] = uci[4]
    return move_obj

# --- Auth Routes ---
@app.route("/signup", methods=["GET", "POST"]) # Add "GET" here
def signup():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return "Username already exists", 400
        
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    
    # If someone tries to visit /signup via GET (typing in URL), send them home
    return redirect(url_for('index'))

@app.route("/login", methods=["GET", "POST"]) # Add "GET" here
def login():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        return "Invalid credentials", 401
    
    # If someone tries to visit /login via GET, send them home
    return redirect(url_for('index'))

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- Game Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_top_moves", methods=["POST"])
def get_top_moves():
    data = request.json
    board = chess.Board(data["fen"])
    top_move_ucis, top_move_probs = model_policy_top_moves(board, top_n=3)
    stockfish_result = get_stockfish_eval(board.fen(), multipv=1)
    board_eval = stockfish_result[0]["score"] if stockfish_result else 0
    move_evals = []
    for i, move_uci in enumerate(top_move_ucis):
        board_copy = chess.Board(board.fen())
        board_copy.push_uci(move_uci)
        eval_result = get_stockfish_eval(board_copy.fen(), multipv=1)
        move_evals.append({"move": move_uci, "score": eval_result[0]["score"] if eval_result else 0, "probability": top_move_probs[i]})
    return jsonify({"top_moves": top_move_ucis, "move_evals": move_evals, "board_eval": board_eval})

@app.route("/ai_move", methods=["POST"])
@login_required
def ai_move():
    board = chess.Board(request.json["fen"])
    top_move_ucis, _ = model_policy_top_moves(board, top_n=3)
    if not top_move_ucis:
        legal_moves = list(board.legal_moves)
        move_uci = random.choice(legal_moves).uci() if legal_moves else None
    else:
        # Use simple weighted random selection
        move_uci = np.random.choice(top_move_ucis, p=[0.8, 0.15, 0.05][:len(top_move_ucis)])
    return jsonify({"move": uci_to_move_obj(move_uci)})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)