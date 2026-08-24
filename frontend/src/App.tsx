import { useState, useCallback } from 'react'
import type { OrchestratorResult, StepRecord, AgentState } from './types/agentic'
import { ImageUploader } from './components/ImageUploader'
import { GoalInput } from './components/GoalInput'
import { OrchestratorTimeline } from './components/OrchestratorTimeline'
import { AgentStatePanel } from './components/AgentStatePanel'
import { VisualOverlay } from './components/VisualOverlay'
import { TranscriptViewer } from './components/TranscriptViewer'
import { FinalAnswer } from './components/FinalAnswer'
import { TVPChainTree } from './components/TVPChainTree'
import { runAgent } from './utils/api'

type TabKey = 'timeline' | 'state' | 'overlay' | 'transcript' | 'tvp' | 'result'

export default function App() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string>('')
  const [goal, setGoal] = useState('detect all objects and count the cars')
  const [maxSteps, setMaxSteps] = useState(12)
  const [maxRetries, setMaxRetries] = useState(1)
  const [activeTab, setActiveTab] = useState<TabKey>('timeline')

  const [result, setResult] = useState<OrchestratorResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  const handleImageChange = useCallback((file: File | null) => {
    setImageFile(file)
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target?.result as string)
      reader.readAsDataURL(file)
    } else {
      setImagePreview('')
    }
  }, [])

  const handleRun = useCallback(async () => {
    if (!imageFile || !goal.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await runAgent({ image: imageFile, goal, max_steps: maxSteps, max_retries: maxRetries })
      setResult(res)
      setActiveTab('timeline')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [imageFile, goal, maxSteps, maxRetries])

  const handleClear = useCallback(() => {
    setImageFile(null)
    setImagePreview('')
    setResult(null)
    setError('')
  }, [])

  const steps: StepRecord[] = result?.steps ?? []
  const agentState: AgentState = result?.state ?? {
    detected_objects: [],
    extracted_text: [],
    located_regions: [],
    counts: {},
    descriptions: [],
    pending_issues: [],
  }

  const tabs: { key: TabKey; label: string }[] = [
    { key: 'timeline', label: '步骤时间轴' },
    { key: 'state', label: 'AgentState' },
    { key: 'overlay', label: '图像叠加' },
    { key: 'transcript', label: 'Transcript' },
    { key: 'tvp', label: 'TVP 推理链' },
    { key: 'result', label: '最终结果' },
  ]

  return (
    <div className="min-h-screen bg-warm-50">
      {/* Header */}
      <header className="bg-white border-b border-warm-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary-600 flex items-center justify-center text-white font-bold text-lg">
              FF
            </div>
            <div>
              <h1 className="text-lg font-semibold text-warm-900">FlorenceForge</h1>
              <p className="text-xs text-warm-500">Agentic 多步视觉推理</p>
            </div>
          </div>
          <div className="text-xs text-warm-400">
            v1.0.0
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Panel — Controls */}
          <div className="lg:col-span-4 space-y-5">
            <div className="panel">
              <h2 className="text-sm font-semibold text-warm-700 mb-3 uppercase tracking-wide">
                输入
              </h2>
              <ImageUploader onImageChange={handleImageChange} />
              <div className="mt-4">
                <GoalInput value={goal} onChange={setGoal} />
              </div>
            </div>

            <div className="panel">
              <h2 className="text-sm font-semibold text-warm-700 mb-3 uppercase tracking-wide">
                配置
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="text-xs text-warm-500 block mb-1">最大步骤数</label>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    value={maxSteps}
                    onChange={(e) => setMaxSteps(Number(e.target.value))}
                    className="w-full accent-primary-600"
                  />
                  <div className="text-right text-xs text-warm-500">{maxSteps}</div>
                </div>
                <div>
                  <label className="text-xs text-warm-500 block mb-1">最大重试次数</label>
                  <input
                    type="range"
                    min={0}
                    max={5}
                    value={maxRetries}
                    onChange={(e) => setMaxRetries(Number(e.target.value))}
                    className="w-full accent-primary-600"
                  />
                  <div className="text-right text-xs text-warm-500">{maxRetries}</div>
                </div>
              </div>

              <div className="mt-5 flex gap-2">
                <button
                  onClick={handleRun}
                  disabled={!imageFile || loading}
                  className="btn-primary flex-1"
                >
                  {loading ? '⏳ 推理中...' : '🚀 运行'}
                </button>
                <button onClick={handleClear} className="btn-secondary">
                  🗑️ 清空
                </button>
              </div>

              {error && (
                <div className="mt-3 p-3 bg-rose-50 text-rose-700 text-sm rounded-lg border border-rose-200">
                  {error}
                </div>
              )}
            </div>

            {/* Mini preview */}
            {imagePreview && (
              <div className="panel">
                <h2 className="text-sm font-semibold text-warm-700 mb-2">图像预览</h2>
                <img
                  src={imagePreview}
                  alt="preview"
                  className="w-full rounded-lg border border-warm-200"
                />
              </div>
            )}
          </div>

          {/* Right Panel — Results */}
          <div className="lg:col-span-8 space-y-5">
            {result ? (
              <>
                {/* Tab Bar */}
                <div className="flex flex-wrap gap-1 bg-warm-100 p-1 rounded-lg">
                  {tabs.map((t) => (
                    <button
                      key={t.key}
                      onClick={() => setActiveTab(t.key)}
                      className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                        activeTab === t.key
                          ? 'bg-white text-warm-900 shadow-sm font-medium'
                          : 'text-warm-500 hover:text-warm-700'
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>

                {/* Tab Content */}
                {activeTab === 'timeline' && (
                  <div className="panel">
                    <OrchestratorTimeline
                      steps={steps}
                      activeStepIndex={steps.length - 1}
                    />
                  </div>
                )}

                {activeTab === 'state' && (
                  <div className="panel">
                    <AgentStatePanel state={agentState} />
                  </div>
                )}

                {activeTab === 'overlay' && imagePreview && (
                  <div className="panel">
                    <VisualOverlay
                      imageSrc={imagePreview}
                      step={steps[steps.length - 1]}
                    />
                  </div>
                )}

                {activeTab === 'transcript' && (
                  <div className="panel">
                    <TranscriptViewer transcript={result.transcript} />
                  </div>
                )}

                {activeTab === 'tvp' && (
                  <div className="panel space-y-4">
                    <TVPChainTree transcript={result.transcript} />
                  </div>
                )}

                {activeTab === 'result' && (
                  <div className="panel">
                    <FinalAnswer
                      answer={result.final_answer}
                      success={result.success}
                      state={agentState}
                    />
                  </div>
                )}
              </>
            ) : (
              <div className="panel min-h-[400px] flex flex-col items-center justify-center text-warm-400">
                <div className="text-4xl mb-3">🎨</div>
                <p className="text-lg font-medium text-warm-600 mb-1">上传图像并输入目标</p>
                <p className="text-sm text-warm-400">系统将自动分解任务、调用工具并验证结果</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
