"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"

export default function RetrievalPage() {
  const [dataset, setDataset] = useState("scifact")
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  
  const handleCompare = async () => {
    setLoading(true)
    try {
      const res = await fetch("/api/v1/retrieval/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, dataset_id: dataset, top_k: 5 })
      })
      if (!res.ok) throw new Error("Failed to fetch")
      const data = await res.json()
      setResults(data)
    } catch (err) {
      console.error(err)
      alert("Error fetching retrieval comparison")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto py-10 space-y-8">
      <div className="flex flex-col gap-2">
        <h1 className="text-4xl font-bold">Retrieval Comparison</h1>
        <p className="text-muted-foreground">Compare BM25, Dense, and Hybrid retrieval strategies.</p>
      </div>

      <Card className="p-6">
        <div className="mb-4">
          <h2 className="text-lg font-semibold">Configuration</h2>
          <p className="text-sm text-muted-foreground">Select dataset and enter a query.</p>
        </div>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Dataset</label>
              <select 
                className="w-full flex h-10 items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={dataset} 
                onChange={(e: any) => setDataset(e.target.value)}
              >
                <option value="scifact">SciFact</option>
                <option value="covidqa" disabled>COVID-QA (Not implemented)</option>
                <option value="hotpotqa" disabled>HotpotQA (Not implemented)</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Query</label>
              <input 
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={query} 
                onChange={(e: any) => setQuery(e.target.value)} 
                placeholder="Enter search query..." 
              />
            </div>
          </div>
          <Button onClick={handleCompare} disabled={loading || !query}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Compare Retrievers
          </Button>
        </div>
      </Card>

      {results && (
        <div className="grid grid-cols-3 gap-6">
          {["bm25", "dense", "hybrid"].map((method) => (
            <Card key={method} className="flex flex-col h-full p-6">
              <div className="mb-4">
                <h2 className="text-lg font-semibold capitalize">{method} Retrieval</h2>
                <p className="text-sm text-muted-foreground">{results[method].latency_ms.toFixed(2)} ms</p>
              </div>
              <div className="flex-1 overflow-auto space-y-4">
                {results[method].results.map((res: any, idx: number) => (
                  <div key={idx} className="border p-3 rounded-md text-sm space-y-1 bg-muted/30">
                    <div className="flex justify-between font-semibold">
                      <span>Rank {idx + 1}</span>
                      <span>Score: {res.score.toFixed(4)}</span>
                    </div>
                    <div className="text-xs text-muted-foreground break-all">{res.doc_id}</div>
                    <div className="line-clamp-4">{res.text}</div>
                  </div>
                ))}
                {results[method].results.length === 0 && <p className="text-sm text-muted-foreground">No results.</p>}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
