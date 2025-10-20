import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import rag

import db


app = Flask(__name__)
CORS(app)

print("🔄 Checking database...")
try:
    db.ensure_tables_exist()
except Exception as e:
    print(f"⚠️ Database initialization error: {e}")

print("🔄 Initializing RAG system...", flush=True)



##endpoints
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Health Assistant API is running"})

@app.route("/question", methods=["POST"])
def handle_question():
    print("\n" + "=" * 60, flush=True)
    print("📨 /question endpoint called", flush=True)

    data = request.json
    question = data.get("question")

    print(f"❓ Question received: {question}", flush=True)

    if not question:
        print("❌ No question provided", flush=True)
        return jsonify({"error": "No question provided"}), 400

    conversation_id = str(uuid.uuid4())
    print(f"🆔 Conversation ID: {conversation_id}", flush=True)

    try:
        print(f"🤔 Calling rag() with question: {question}", flush=True)
        answer_data = rag(question)
        print(f"✅ rag() returned successfully", flush=True)
        print(f"💬 Answer: {answer_data['answer'][:100]}...", flush=True)
    except Exception as e:
        print(f"❌ Error in rag(): {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    result = {
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer_data["answer"],
    }


@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]

    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400

    db.save_feedback(
        conversation_id=conversation_id,
        feedback=feedback,
    )

    result = {
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5050)