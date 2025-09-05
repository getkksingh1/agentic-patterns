import Link from 'next/link'
import { ArrowRight, Book, Code, Users, Shield, Zap, Target, CheckCircle } from 'lucide-react'

export default function Hero() {
  return (
    <section className="section bg-gradient-to-br from-primary-50 to-blue-50 relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary-200 rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-blue-200 rounded-full blur-3xl"></div>
      </div>
      
      <div className="container relative z-10">
        <div className="text-center max-w-6xl mx-auto">
          <h1 className="text-5xl lg:text-6xl font-bold mb-6">
            The Complete Guide to{' '}
            <span className="text-primary-600">Agentic Patterns</span>
          </h1>
          
          {/* Enhanced Value Proposition */}
          <div className="mb-8">
            <p className="text-2xl lg:text-3xl font-semibold text-gray-800 mb-4">
              Master 21 Proven Patterns to Build Scalable, Safe, and Human-Aligned AI Systems
            </p>
            <p className="text-xl text-gray-600 leading-relaxed max-w-4xl mx-auto">
              From prompt chaining to multi-agent orchestration, learn the essential patterns that power 
              production-ready autonomous AI agents with comprehensive code examples and real-world implementations.
            </p>
          </div>
          
          {/* Enhanced CTAs */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
            <Link href="/table-of-contents" className="bg-primary-600 hover:bg-primary-700 text-white px-8 py-4 rounded-lg font-semibold inline-flex items-center justify-center text-lg shadow-lg hover:shadow-xl transition-all duration-200 transform hover:-translate-y-1">
              <Zap className="mr-3" size={24} />
              Start Learning Now
              <ArrowRight className="ml-3" size={24} />
            </Link>
            <Link href="/about" className="bg-white hover:bg-gray-50 text-gray-700 px-8 py-4 rounded-lg font-medium inline-flex items-center justify-center text-lg border border-gray-200 hover:border-gray-300 transition-all duration-200">
              <Book className="mr-3" size={20} />
              Learn More
            </Link>
          </div>
          
          {/* Value Points */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center mb-16 text-sm">
            <div className="flex items-center gap-2 text-gray-600">
              <CheckCircle className="text-green-500" size={16} />
              21 comprehensive chapters
            </div>
            <div className="flex items-center gap-2 text-gray-600">
              <CheckCircle className="text-green-500" size={16} />
              Production-ready code examples
            </div>
            <div className="flex items-center gap-2 text-gray-600">
              <CheckCircle className="text-green-500" size={16} />
              Real-world implementations
            </div>
          </div>

          {/* Visual Diagram */}
          <div className="mb-16 bg-white/80 backdrop-blur-sm rounded-2xl p-8 border border-white/50 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-6">AI Agent Architecture Flow</h3>
            <div className="flex flex-col lg:flex-row items-center justify-center gap-8 text-center">
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mb-3 shadow-lg">
                  <Target className="text-white" size={28} />
                </div>
                <h4 className="font-semibold text-gray-800">Input & Planning</h4>
                <p className="text-sm text-gray-600 mt-1">Goal setting, context analysis</p>
              </div>
              
              <ArrowRight className="text-gray-400 rotate-90 lg:rotate-0" size={24} />
              
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center mb-3 shadow-lg">
                  <Code className="text-white" size={28} />
                </div>
                <h4 className="font-semibold text-gray-800">Processing & Reasoning</h4>
                <p className="text-sm text-gray-600 mt-1">Pattern execution, tool use</p>
              </div>
              
              <ArrowRight className="text-gray-400 rotate-90 lg:rotate-0" size={24} />
              
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mb-3 shadow-lg">
                  <Users className="text-white" size={28} />
                </div>
                <h4 className="font-semibold text-gray-800">Human Collaboration</h4>
                <p className="text-sm text-gray-600 mt-1">Feedback, safety, alignment</p>
              </div>
              
              <ArrowRight className="text-gray-400 rotate-90 lg:rotate-0" size={24} />
              
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-red-600 rounded-xl flex items-center justify-center mb-3 shadow-lg">
                  <Shield className="text-white" size={28} />
                </div>
                <h4 className="font-semibold text-gray-800">Scaling & Safety</h4>
                <p className="text-sm text-gray-600 mt-1">Monitoring, optimization</p>
              </div>
            </div>
          </div>

          {/* Enhanced Guide Overview */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6 text-center hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Book className="text-white" size={28} />
              </div>
              <h3 className="text-xl font-bold mb-2 text-green-800">Part One</h3>
              <p className="text-green-700 font-medium text-sm mb-3">
                Foundations of Agentic Patterns
              </p>
              <div className="flex items-center justify-center gap-2 text-xs text-green-600 bg-green-100 px-3 py-1 rounded-full">
                <CheckCircle size={14} />
                7 Chapters
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-6 text-center hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Code className="text-white" size={28} />
              </div>
              <h3 className="text-xl font-bold mb-2 text-blue-800">Part Two</h3>
              <p className="text-blue-700 font-medium text-sm mb-3">
                Learning and Adaptation
              </p>
              <div className="flex items-center justify-center gap-2 text-xs text-blue-600 bg-blue-100 px-3 py-1 rounded-full">
                <CheckCircle size={14} />
                4 Chapters
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6 text-center hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Users className="text-white" size={28} />
              </div>
              <h3 className="text-xl font-bold mb-2 text-purple-800">Part Three</h3>
              <p className="text-purple-700 font-medium text-sm mb-3">
                Human-Centric Patterns
              </p>
              <div className="flex items-center justify-center gap-2 text-xs text-purple-600 bg-purple-100 px-3 py-1 rounded-full">
                <CheckCircle size={14} />
                3 Chapters
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-red-50 to-pink-50 border border-red-200 rounded-xl p-6 text-center hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-pink-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <Shield className="text-white" size={28} />
              </div>
              <h3 className="text-xl font-bold mb-2 text-red-800">Part Four</h3>
              <p className="text-red-700 font-medium text-sm mb-3">
                Scaling, Safety & Discovery
              </p>
              <div className="flex items-center justify-center gap-2 text-xs text-red-600 bg-red-100 px-3 py-1 rounded-full">
                <CheckCircle size={14} />
                7 Chapters
              </div>
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
