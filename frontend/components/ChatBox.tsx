"use client";
import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot } from 'lucide-react';
import { ThinkingAccordion } from './ThinkingAccordion';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    steps?: string[];
}

interface ChatBoxProps {
    messages: Message[];
    loading: boolean;
}

export const ChatBox: React.FC<ChatBoxProps> = ({ messages, loading }) => {
    return (
        <div className="flex-1 overflow-y-auto space-y-6 p-4">
            {messages.map((msg, idx) => (
                <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center shrink-0">
                            <Bot className="w-5 h-5 text-white" />
                        </div>
                    )}

                    <div className={`max-w-2xl space-y-2 ${msg.role === 'user' ? 'items-end flex flex-col' : ''}`}>
                        {msg.steps && msg.steps.length > 0 && <ThinkingAccordion steps={msg.steps} />}

                        <div className={`p-4 rounded-2xl text-sm leading-relaxed ${msg.role === 'user'
                            ? 'bg-blue-600 text-white rounded-br-none'
                            : 'bg-gray-800 text-gray-200 rounded-bl-none shadow-sm'
                            }`}>
                            <div className="prose prose-invert max-w-none prose-sm">
                                <ReactMarkdown>
                                    {msg.content}
                                </ReactMarkdown>
                            </div>
                        </div>
                    </div>

                    {msg.role === 'user' && (
                        <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center shrink-0">
                            <User className="w-5 h-5 text-white" />
                        </div>
                    )}
                </div>
            ))}
            {loading && (
                <div className="flex gap-4">
                    <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center shrink-0">
                        <Bot className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex items-center gap-1 h-8">
                        <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-0"></span>
                        <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-150"></span>
                        <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-300"></span>
                    </div>
                </div>
            )}
        </div>
    );
};
