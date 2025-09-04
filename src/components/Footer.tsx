import Link from 'next/link'

export default function Footer() {
  const currentYear = new Date().getFullYear()

  const footerLinks = {
    'Resources': [
      { name: 'All Patterns', href: '/patterns' },
      { name: 'About', href: '/about' },
    ],
    'Community': [
      { name: 'GitHub', href: 'https://github.com' },
      { name: 'Discord', href: '#' },
      { name: 'Twitter', href: '#' },
    ],
  }

  return (
    <footer className="bg-gray-900 text-white">
      <div className="container py-12">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <Link href="/" className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">A</span>
              </div>
              <span className="text-xl font-bold">Agentic Patterns</span>
            </Link>
            <p className="text-gray-400 max-w-md">
              Your comprehensive resource for understanding and implementing autonomous AI agent 
              design patterns and architectures.
            </p>
          </div>

          {/* Footer Links */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h3 className="font-semibold mb-4">{category}</h3>
              <ul className="space-y-2">
                {links.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      className="text-gray-400 hover:text-white transition-colors"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
          <p>&copy; {currentYear} Agentic Patterns. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
