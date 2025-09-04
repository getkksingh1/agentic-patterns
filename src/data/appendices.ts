import { Appendix } from './types'
import { appendixAPrompting } from './appendices/appendix-a-advanced-prompting'
import { appendixBAIInteractions } from './appendices/appendix-b-ai-interactions'
import { appendixCAgenticFrameworks } from './appendices/appendix-c-agentic-frameworks'
import { appendixDAgentSpace } from './appendices/appendix-d-agentspace'
import { appendixEAIAgentsCLI } from './appendices/appendix-e-ai-agents-cli'
import { appendixFUnderTheHood } from './appendices/appendix-f-under-the-hood'
import { appendixGCodingAgents } from './appendices/appendix-g-coding-agents'

const appendices: Appendix[] = [
  appendixAPrompting,
  appendixBAIInteractions,
  appendixCAgenticFrameworks,
  appendixDAgentSpace,
  appendixEAIAgentsCLI,
  appendixFUnderTheHood,
  appendixGCodingAgents,
]

export const getAppendixBySlug = (slug: string): Appendix | undefined => {
  return appendices.find(appendix => appendix.id === slug)
}

export const getAllAppendices = (): Appendix[] => {
  return appendices
}
