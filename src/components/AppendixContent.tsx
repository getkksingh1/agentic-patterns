import { ArrowLeft, FileText, Globe } from 'lucide-react'
import Link from 'next/link'

interface Appendix {
  id: string
  letter: string
  title: string
  description: string
  onlineOnly?: boolean
  content: {
    overview: string
    sections: {
      title: string
      content: string
    }[]
    resources?: string[]
  }
}

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
            <span className="text-primary-600 font-medium">Appendix {appendix.letter}</span>
            {appendix.onlineOnly && (
              <>
                <span className="text-gray-300">•</span>
                <div className="flex items-center gap-2 text-primary-600">
                  <Globe size={16} />
                  <span className="text-sm">Online Only</span>
                </div>
              </>
            )}
          </div>
          
          <h1 className="text-4xl lg:text-5xl font-bold mb-4">
            {appendix.title}
          </h1>
          
          <p className="text-xl text-gray-600 mb-6">
            {appendix.description}
          </p>
        </div>

        {/* Appendix Content */}
        <div className="prose prose-lg max-w-none">
          {/* Overview */}
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <FileText size={20} />
              Overview
            </h2>
            <p className="text-gray-600 leading-relaxed">
              {appendix.content.overview}
            </p>
          </section>

          {/* Sections */}
          {appendix.content.sections.map((section, index) => (
            <section key={index} className="mb-8">
              <h2 className="text-2xl font-semibold mb-4">{section.title}</h2>
              <div className="text-gray-600 leading-relaxed whitespace-pre-line">
                {section.content}
              </div>
            </section>
          ))}

          {/* Resources */}
          {appendix.content.resources && (
            <section className="mb-8">
              <h2 className="text-2xl font-semibold mb-4">Additional Resources</h2>
              <ul className="space-y-2">
                {appendix.content.resources.map((resource, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <span className="text-primary-500 mt-2">•</span>
                    <span className="text-gray-600">{resource}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
