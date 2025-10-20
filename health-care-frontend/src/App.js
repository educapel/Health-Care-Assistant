import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:5050';

function App() {
  const [question, setQuestion] = useState('');
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/question`, {
        question: question
      });

      const newConversation = {
        id: response.data.conversation_id,
        question: response.data.question,
        answer: response.data.answer,
        feedback: null,
        timestamp: new Date().toLocaleTimeString()
      };

      setConversations([newConversation, ...conversations]);
      setQuestion('');
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to connect to server');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (conversationId, feedbackValue) => {
    try {
      await axios.post(`${API_BASE_URL}/feedback`, {
        conversation_id: conversationId,
        feedback: feedbackValue
      });

      setConversations(conversations.map(conv =>
        conv.id === conversationId
          ? { ...conv, feedback: feedbackValue }
          : conv
      ));
    } catch (err) {
      console.error('Error submitting feedback:', err);
      alert('Failed to submit feedback');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏥 Health Care Assistant</h1>
        <p>Ask me anything about health and medical topics</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="question-form">
          <div className="input-group">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask your health question..."
              disabled={loading}
              className="question-input"
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="submit-button"
            >
              {loading ? '⏳ Thinking...' : '🚀 Ask'}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message">
            ❌ {error}
            <button onClick={() => setError(null)} className="close-error">×</button>
          </div>
        )}

        <div className="conversations">
          {conversations.map((conv) => (
            <div key={conv.id} className="conversation-card">
              <div className="conversation-header">
                <span className="timestamp">{conv.timestamp}</span>
              </div>
              <div className="question-section">
                <strong>❓ Question:</strong> {conv.question}
              </div>
              <div className="answer-section">
                <strong>💡 Answer:</strong> {conv.answer}
              </div>
              <div className="feedback-section">
                <span>Was this helpful?</span>
                <button
                  onClick={() => handleFeedback(conv.id, 1)}
                  className={`feedback-btn ${conv.feedback === 1 ? 'active-positive' : ''}`}
                  disabled={conv.feedback !== null}
                >
                  👍 Helpful
                </button>
                <button
                  onClick={() => handleFeedback(conv.id, -1)}
                  className={`feedback-btn ${conv.feedback === -1 ? 'active-negative' : ''}`}
                  disabled={conv.feedback !== null}
                >
                  👎 Not Helpful
                </button>
                {conv.feedback && (
                  <span className="feedback-thanks">Thanks for your feedback!</span>
                )}
              </div>
            </div>
          ))}
        </div>

        {conversations.length === 0 && !loading && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <p>No conversations yet.</p>
            <p>Ask your first health question above!</p>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>⚠️ This is an AI assistant. Always consult healthcare professionals for medical advice.</p>
      </footer>
    </div>
  );
}

export default App;
