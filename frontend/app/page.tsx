"use client";
import React, { useState, useRef } from 'react';
import { Send, Sparkles } from 'lucide-react';
import { ChatBox } from '../components/ChatBox';
import { PDFUploader } from '../components/PDFUploader';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  steps?: string[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    // Initial bot message placeholder
    const botMessageId = Date.now();
    setMessages(prev => [...prev, { role: 'assistant', content: '', steps: [] }]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage.content }),
      });

      if (!response.ok) throw new Error('Chat request failed');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) return;

      let currentContent = '';
      let currentSteps: string[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') break;

            try {
              const data = JSON.parse(dataStr);

              if (data.type === 'token') {
                currentContent += data.content;
              } else if (data.type === 'step') {
                currentSteps = [...currentSteps, data.content];
              }

              // Update state
              setMessages(prev => {
                const newMessages = [...prev];
                const lastMsg = newMessages[newMessages.length - 1];
                if (lastMsg.role === 'assistant') {
                  lastMsg.content = currentContent;
                  lastMsg.steps = currentSteps;
                }
                return newMessages;
              });

              scrollToBottom();

            } catch (e) {
              console.error("Error parsing SSE:", e);
            }
          }
        }
      }

    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, something went wrong.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex h-screen bg-gray-950 text-gray-100 font-sans selection:bg-purple-500/30">
      {/* Sidebar */}
      <div className="w-80 border-r border-gray-800 bg-gray-900/50 p-6 flex flex-col gap-6 hidden md:flex">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-2 bg-gradient-to-tr from-purple-500 to-blue-500 rounded-lg">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-blue-400">
            Shadow CFO
          </h1>
        </div>

        <div className="space-y-4">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Data Source</h2>
          <PDFUploader />
        </div>

        <div className="mt-auto">
          <div className="p-4 rounded-xl bg-gray-800/50 border border-gray-700/50">
            <h3 className="text-sm font-medium text-gray-300 mb-1">Agent Capability</h3>
            <p className="text-xs text-gray-500 leading-relaxed">
              Powered by <strong>LangGraph</strong>.
              Capable of multi-step reasoning, hybrid search, and financial verification.
            </p>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full bg-gradient-to-b from-gray-900 to-gray-950">
        <header className="p-4 border-b border-gray-800 md:hidden block">
          <h1 className="text-lg font-bold">Shadow CFO</h1>
        </header>

        <ChatBox messages={messages} loading={loading} />
        <div ref={messagesEndRef} />

        <div className="p-4 border-t border-gray-800 bg-gray-900/80 backdrop-blur-sm">
          <div className="max-w-3xl mx-auto relative">
            <input
              className="w-full bg-gray-800 border-none rounded-2xl py-4 pl-6 pr-14 text-white placeholder-gray-500 focus:ring-2 focus:ring-purple-500/50 focus:outline-none transition-all shadow-lg"
              placeholder="Ask a financial question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-purple-600 rounded-xl hover:bg-purple-500 disabled:opacity-50 disabled:hover:bg-purple-600 transition-colors"
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>
          <p className="text-center text-xs text-gray-600 mt-3">
            AI can make mistakes. Please verify important financial data.
          </p>
        </div>
      </div>
    </main>
  );
}
