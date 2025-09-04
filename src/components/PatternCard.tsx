interface Pattern {
  id: string
  title: string
  description: string
  category: string
  difficulty: string
  applications: string[]
}

interface PatternCardProps {
  pattern: Pattern
}

export default function PatternCard({ pattern }: PatternCardProps) {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return 'bg-green-100 text-green-800'
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-800'
      case 'advanced':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="card hover:shadow-lg transition-shadow group">
      <div className="flex justify-between items-start mb-3">
        <span className="text-sm font-medium text-primary-600 bg-primary-50 px-3 py-1 rounded-full">
          {pattern.category}
        </span>
        <span className={`text-xs px-2 py-1 rounded-full ${getDifficultyColor(pattern.difficulty)}`}>
          {pattern.difficulty}
        </span>
      </div>
      
      <h3 className="text-xl font-semibold mb-3 group-hover:text-primary-600 transition-colors">
        {pattern.title}
      </h3>
      
      <p className="text-gray-600 mb-4 line-clamp-3">
        {pattern.description}
      </p>
      
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Applications:</h4>
        <div className="flex flex-wrap gap-2">
          {pattern.applications.map((app) => (
            <span
              key={app}
              className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded"
            >
              {app}
            </span>
          ))}
        </div>
      </div>
      
      <div className="flex justify-between items-center pt-4 border-t border-gray-100">
        <span className="text-sm text-gray-500">Learn more</span>
        <span className="text-primary-600 group-hover:text-primary-700 transition-colors">
          →
        </span>
      </div>
    </div>
  )
}
