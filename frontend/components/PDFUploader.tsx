"use client";
import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

export const PDFUploader = () => {
    const [uploading, setUploading] = useState(false);
    const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
    const [message, setMessage] = useState('');

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.[0]) return;

        const file = e.target.files[0];
        setUploading(true);
        setStatus('idle');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error('Upload failed');

            const data = await res.json();
            setStatus('success');
            setMessage(`Indexed ${data.chunks_processed} chunks successfully.`);
        } catch (err) {
            setStatus('error');
            setMessage('Failed to upload PDF.');
            console.error(err);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="p-4 border border-dashed border-gray-700 rounded-xl bg-gray-900/50 hover:bg-gray-900/80 transition-colors">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-500/10 rounded-lg">
                        <FileText className="w-6 h-6 text-blue-400" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-gray-200">Upload 10-K Document</h3>
                        <p className="text-xs text-gray-500">Support for PDF files only</p>
                    </div>
                </div>
                <div>
                    <input
                        type="file"
                        id="pdf-upload"
                        className="hidden"
                        accept=".pdf"
                        onChange={handleUpload}
                        disabled={uploading}
                    />
                    <label
                        htmlFor="pdf-upload"
                        className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-md cursor-pointer transition-colors ${uploading ? 'bg-gray-700 text-gray-400 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-500'
                            }`}
                    >
                        {uploading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
                        {uploading ? 'Uploading...' : 'Select File'}
                    </label>
                </div>
            </div>
            {(status === 'success' || status === 'error') && (
                <div className={`mt-3 flex items-center gap-2 text-xs ${status === 'success' ? 'text-green-400' : 'text-red-400'}`}>
                    {status === 'success' ? <CheckCircle className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
                    {message}
                </div>
            )}
        </div>
    );
};
