import Link from 'next/link'
import { ArrowRight, Book, Code, Users, Shield } from 'lucide-react'

export default function Hero() {
  return (
    <section className="section bg-gradient-to-br from-primary-50 to-blue-50">
      <div className="container">
        <div className="text-center max-w-5xl mx-auto">
          <h1 className="text-5xl lg:text-6xl font-bold mb-6">
            The Complete Guide to{' '}
            <span className="text-primary-600">Agentic Patterns</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            A comprehensive resource covering 21 essential patterns for building autonomous AI agents, 
            from foundational techniques to advanced scaling and safety considerations.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <Link href="/table-of-contents" className="btn-primary inline-flex items-center">
              View Full Guide
              <Book className="ml-2" size={20} />
            </Link>
            <Link href="/about" className="btn-secondary">
              Learn More
            </Link>
          </div>

          {/* Guide Overview */}
          <div className="grid md:grid-cols-4 gap-6 mt-16">
            <div className="card text-center">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Book className="text-green-600" size={24} />
              </div>
              <h3 className="text-lg font-semibold mb-2">Part One</h3>
              <p className="text-gray-600 text-sm">
                Foundations of Agentic Patterns
              </p>
              <p className="text-xs text-gray-500 mt-2">7 Chapters</p>
            </div>
            
            <div className="card text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Code className="text-blue-600" size={24} />
              </div>
              <h3 className="text-lg font-semibold mb-2">Part Two</h3>
              <p className="text-gray-600 text-sm">
                Learning and Adaptation
              </p>
              <p className="text-xs text-gray-500 mt-2">4 Chapters</p>
            </div>
            
            <div className="card text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Users className="text-purple-600" size={24} />
              </div>
              <h3 className="text-lg font-semibold mb-2">Part Three</h3>
              <p className="text-gray-600 text-sm">
                Human-Centric Patterns
              </p>
              <p className="text-xs text-gray-500 mt-2">3 Chapters</p>
            </div>
            
            <div className="card text-center">
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Shield className="text-red-600" size={24} />
              </div>
              <h3 className="text-lg font-semibold mb-2">Part Four</h3>
              <p className="text-gray-600 text-sm">
                Scaling, Safety & Discovery
              </p>
              <p className="text-xs text-gray-500 mt-2">7 Chapters</p>
            </div>
          </div>

          <div className="mt-12 p-6 bg-white rounded-xl border border-gray-200">
            <p className="text-gray-600">
              <strong>21 Comprehensive Chapters</strong> covering everything from basic prompt chaining 
              to advanced multi-agent systems, plus <strong>7 detailed appendices</strong> with practical resources and implementations.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
