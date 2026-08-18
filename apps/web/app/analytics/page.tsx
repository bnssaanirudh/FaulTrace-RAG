"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Loader2 } from "lucide-react"

export default function AnalyticsPage() {
  const [dataset, setDataset] = useState("scifact")
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  
  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true)
      try {
        const res = await fetch(`/api/v1/analytics/${dataset}`)
        if (!res.ok) throw new Error("Failed to fetch")
        const json = await res.json()
        setData(json)
      } catch (err) {
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    
    fetchAnalytics()
  }, [dataset])

  return (
    <div className="container mx-auto py-10 space-y-8">
      <div className="flex flex-col gap-2">
        <h1 className="text-4xl font-bold">Corpus Analytics</h1>
        <p className="text-muted-foreground">Exploratory text mining on benchmark datasets.</p>
      </div>
      
      <div className="flex items-center gap-4">
        <span className="font-semibold">Dataset:</span>
        <select 
          className="flex h-10 items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm w-[200px]"
          value={dataset} 
          onChange={(e: any) => setDataset(e.target.value)}
        >
          <option value="scifact">SciFact</option>
        </select>
      </div>
      
      {loading && (
        <div className="flex justify-center p-12">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      )}
      
      {!loading && data && (
        <div className="grid grid-cols-2 gap-6">
          <Card className="p-6">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Corpus Summary</h2>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Total Documents:</span>
                <span className="font-semibold">{data.summary.doc_count}</span>
              </div>
              <div className="flex justify-between">
                <span>Total Words:</span>
                <span className="font-semibold">{data.summary.total_words}</span>
              </div>
              <div className="flex justify-between">
                <span>Average Length (words):</span>
                <span className="font-semibold">{Math.round(data.summary.avg_length)}</span>
              </div>
              <div className="flex justify-between">
                <span>Max Length:</span>
                <span className="font-semibold">{data.summary.max_length}</span>
              </div>
            </div>
          </Card>
          
          <Card className="p-6">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Top TF-IDF Terms</h2>
            </div>
            <div>
              <div className="flex flex-wrap gap-2">
                {data.top_terms.map((t: any, i: number) => (
                  <span key={i} className="px-2 py-1 bg-secondary text-secondary-foreground rounded-md text-xs">
                    {t.term} ({t.score.toFixed(1)})
                  </span>
                ))}
              </div>
            </div>
          </Card>
          
          <Card className="p-6">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Top Bigrams</h2>
            </div>
            <div className="space-y-2">
              {data.top_bigrams.map((b: any, i: number) => (
                <div key={i} className="flex justify-between text-sm">
                  <span>{b.bigram}</span>
                  <span className="text-muted-foreground">{b.count}</span>
                </div>
              ))}
            </div>
          </Card>
          
          <Card className="p-6">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Named Entity Co-occurrence</h2>
            </div>
            <div className="space-y-2">
              {data.entity_cooccurrences.map((e: any, i: number) => (
                <div key={i} className="flex justify-between text-sm">
                  <span>{e.pair.join(" + ")}</span>
                  <span className="text-muted-foreground">{e.count}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
