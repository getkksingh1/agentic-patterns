// Import types and individual chapters
import { Chapter } from './types'
import {
  promptChainingChapter,
  routingChapter,
  parallelizationChapter,
  reflectionChapter,
  toolUseChapter,
  planningChapter,
  multiAgentChapter,
  memoryManagementChapter,
  learningAdaptationChapter,
  modelContextProtocolChapter,
  goalSettingMonitoringChapter,
  exceptionHandlingChapter,
  humanInLoopChapter,
  knowledgeRetrievalChapter,
  interAgentCommunicationChapter,
  resourceAwareOptimizationChapter,
  reasoningTechniquesChapter,
  guardrailsSafetyPatternsChapter,
  evaluationMonitoringChapter
} from './chapters/index'

const chapters: Chapter[] = [
  promptChainingChapter,
  routingChapter,
  parallelizationChapter,
  reflectionChapter,
  toolUseChapter,
  planningChapter,
  multiAgentChapter,
  memoryManagementChapter,
  learningAdaptationChapter,
  modelContextProtocolChapter,
  goalSettingMonitoringChapter,
  exceptionHandlingChapter,
  humanInLoopChapter,
  knowledgeRetrievalChapter,
  interAgentCommunicationChapter,
  resourceAwareOptimizationChapter,
  reasoningTechniquesChapter,
  guardrailsSafetyPatternsChapter,
  evaluationMonitoringChapter
  // Add more chapters as needed...
]

export const getChapterBySlug = (slug: string): Chapter | undefined => {
  return chapters.find(chapter => chapter.id === slug)
}

export const getAllChapters = (): Chapter[] => {
  return chapters
}

export const getChaptersByPart = (part: string): Chapter[] => {
  return chapters.filter(chapter => chapter.part?.includes(part))
}
