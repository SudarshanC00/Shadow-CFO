"use client";
import React, { useState } from 'react';
import { ChevronDown, ChevronRight, BrainCircuit } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface ThinkingAccordionProps {
    steps: string[];
}

export const ThinkingAccordion: React.FC<ThinkingAccordionProps> = ({ steps }) => {
    const [isOpen, setIsOpen] = useState(true);

    if (steps.length === 0) return null;

    return (
        <div className="w-full my-2 border border-gray-700/50 rounded-lg overflow-hidden bg-gray-900/30">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-3 bg-gray-800/50 hover:bg-gray-800/70 transition-colors text-sm font-medium text-gray-300"
            >
                <div className="flex items-center gap-2">
                    <BrainCircuit className="w-4 h-4 text-purple-400" />
                    <span>Thinking Process ({steps.length} steps)</span>
                </div>
                {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-gray-700/50"
                    >
                        <div className="p-3 space-y-2 text-xs font-mono text-gray-400 bg-black/20">
                            {steps.map((step, idx) => (
                                <div key={idx} className="flex gap-2">
                                    <span className="text-gray-600">{(idx + 1).toString().padStart(2, '0')}</span>
                                    <span>{step}</span>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
