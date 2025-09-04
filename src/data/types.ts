// Shared type definitions for chapter data

export interface Chapter {
  id: string
  number?: number
  title: string
  subtitle?: string
  part?: string
  description: string
  readingTime: string
  difficulty?: 'Beginner' | 'Intermediate' | 'Advanced'
  // New structure properties (optional to support both)
  overview?: string
  keyPoints?: string[]
  codeExample?: string
  practicalApplications?: string[]
  nextSteps?: string[]
  sections?: {
    title: string
    content: string
  }[]
  practicalExamples?: {
    title: string
    description: string
    implementation?: string
    example?: string
    steps?: string[]
  }[]
  references?: string[]
  navigation: {
    previous?: { title: string, href: string }
    next?: { title: string, href: string }
  }
  // Legacy support for old structure
  content?: {
    overview: string
    keyPoints: string[]
    codeExample?: string
    practicalApplications: string[]
    nextSteps: string[]
  }
}

export interface ChapterMetadata {
  id: string
  number: number
  title: string
  part: string
}

export interface Appendix {
  id: string
  title: string
  subtitle?: string
  description: string
  readingTime: string
  content: string
}
