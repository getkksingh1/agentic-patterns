import { Code, Users, Lightbulb, Settings } from 'lucide-react'

export default function PatternOverview() {
  const categories = [
    {
      icon: Code,
      title: 'Implementation Patterns',
      description: 'Technical patterns for building and deploying agentic systems',
      count: 12,
    },
    {
      icon: Users,
      title: 'Multi-Agent Patterns',
      description: 'Coordination and communication patterns for agent teams',
      count: 8,
    },
    {
      icon: Lightbulb,
      title: 'Reasoning Patterns',
      description: 'Cognitive architectures for decision-making and problem-solving',
      count: 15,
    },
    {
      icon: Settings,
      title: 'Integration Patterns',
      description: 'Patterns for integrating agents with existing systems and workflows',
      count: 10,
    },
  ]

  return (
    <section className="section bg-white">
      <div className="container">
        <div className="text-center mb-12">
          <h2 className="text-3xl lg:text-4xl font-bold mb-4">
            Pattern Categories
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Our comprehensive collection covers all aspects of agentic system design, 
            from basic implementation to advanced multi-agent coordination.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {categories.map((category) => {
            const IconComponent = category.icon
            return (
              <div key={category.title} className="card hover:shadow-md transition-shadow">
                <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
                  <IconComponent className="text-primary-600" size={24} />
                </div>
                <h3 className="text-lg font-semibold mb-2">{category.title}</h3>
                <p className="text-gray-600 mb-4">{category.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">{category.count} patterns</span>
                  <span className="text-primary-600 text-sm font-medium">Explore →</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
