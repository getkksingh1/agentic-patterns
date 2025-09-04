import Header from '@/components/Header'
import Footer from '@/components/Footer'

export default function AboutPage() {
  return (
    <>
      <Header />
      <main className="section">
        <div className="container">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-4xl lg:text-5xl font-bold mb-8 text-center">
              About Agentic Patterns
            </h1>
            
            <div className="prose prose-lg max-w-none">
              <div className="card mb-8">
                <h2 className="text-2xl font-semibold mb-4">What are Agentic Patterns?</h2>
                <p className="text-gray-600 leading-relaxed">
                  Agentic patterns are design patterns and architectural approaches for building 
                  autonomous AI systems that can operate independently, make decisions, and take 
                  actions to achieve specific goals. These patterns draw from decades of research 
                  in artificial intelligence, multi-agent systems, and autonomous computing.
                </p>
              </div>
              
              <div className="card mb-8">
                <h2 className="text-2xl font-semibold mb-4">Key Characteristics</h2>
                <ul className="space-y-3 text-gray-600">
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2">•</span>
                    <strong>Autonomy:</strong> Agents can operate without constant human supervision
                  </li>
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2">•</span>
                    <strong>Reasoning:</strong> Ability to analyze situations and make informed decisions
                  </li>
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2">•</span>
                    <strong>Planning:</strong> Capacity to create and execute multi-step strategies
                  </li>
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2">•</span>
                    <strong>Adaptability:</strong> Learning and adjusting behavior based on feedback
                  </li>
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2">•</span>
                    <strong>Goal-oriented:</strong> Working towards specific objectives and outcomes
                  </li>
                </ul>
              </div>
              
              <div className="card mb-8">
                <h2 className="text-2xl font-semibold mb-4">Applications</h2>
                <p className="text-gray-600 leading-relaxed mb-4">
                  Agentic patterns are being applied across various domains to create more 
                  intelligent and autonomous systems:
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Business & Operations</h3>
                    <ul className="text-gray-600 space-y-1">
                      <li>• Customer service automation</li>
                      <li>• Process optimization</li>
                      <li>• Decision support systems</li>
                    </ul>
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Technology & Development</h3>
                    <ul className="text-gray-600 space-y-1">
                      <li>• Code generation and review</li>
                      <li>• System monitoring</li>
                      <li>• Automated testing</li>
                    </ul>
                  </div>
                </div>
              </div>
              
              <div className="card">
                <h2 className="text-2xl font-semibold mb-4">Our Mission</h2>
                <p className="text-gray-600 leading-relaxed">
                  This website serves as a comprehensive resource for understanding, implementing, 
                  and advancing agentic patterns. We aim to provide clear explanations, practical 
                  examples, and best practices for building effective autonomous AI systems.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </>
  )
}
