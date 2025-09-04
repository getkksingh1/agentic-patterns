import { ArrowLeft, Clock, FileText } from 'lucide-react'
import Link from 'next/link'
import { Appendix } from '@/data/types'

interface AppendixContentProps {
  appendix: Appendix
}

export default function AppendixContent({ appendix }: AppendixContentProps) {
  return (
    <div className="section bg-white">
      <div className="container max-w-4xl">
        {/* Appendix Header */}
        <div className="mb-8">
          <Link 
            href="/table-of-contents"
            className="inline-flex items-center text-primary-600 hover:text-primary-700 mb-4"
          >
            <ArrowLeft size={16} className="mr-2" />
            Back to Table of Contents
          </Link>
          
          <div className="flex items-center gap-4 mb-4">
            <div className="flex items-center gap-2 text-primary-600">
              <Clock size={16} />
              <span className="text-sm">{appendix.readingTime}</span>
            </div>
          </div>
          
          <h1 className="text-4xl lg:text-5xl font-bold mb-4">
            {appendix.title}
          </h1>
          
          {appendix.subtitle && (
            <h2 className="text-xl lg:text-2xl text-gray-700 mb-4">
              {appendix.subtitle}
            </h2>
          )}
          
          <p className="text-xl text-gray-600 mb-6">
            {appendix.description}
          </p>
        </div>

        {/* Appendix Content */}
        <div className="prose prose-lg max-w-none">
          <div 
            className="text-gray-700 leading-relaxed"
            dangerouslySetInnerHTML={{ 
              __html: appendix.content
                .replace(/```(\w+)?\n([\s\S]*?)\n```/g, '<pre><code class="language-$1">$2</code></pre>')
                .replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1 py-0.5 rounded text-sm">$1</code>')
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                .replace(/\*([^*]+)\*/g, '<em>$1</em>')
                .replace(/^### (.+)$/gm, '<h3 class="text-xl font-semibold mt-8 mb-4">$1</h3>')
                .replace(/^## (.+)$/gm, '<h2 class="text-2xl font-semibold mt-10 mb-6">$1</h2>')
                .replace(/^# (.+)$/gm, '<h1 class="text-3xl font-bold mt-12 mb-8">$1</h1>')
                .replace(/^\- (.+)$/gm, '<li class="mb-2">$1</li>')
                .replace(/^(\d+)\. (.+)$/gm, '<li class="mb-2">$2</li>')
                .replace(/\n\n/g, '</p><p class="mb-4">')
                .replace(/^(?!<[h|l|p|d])/gm, '<p class="mb-4">')
            }}
          />
        </div>
      </div>
    </div>
  )
}
