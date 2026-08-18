"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export default function BenchmarksPage() {
  const benchmarks = [
    {
      id: "scifact",
      title: "SciFact",
      description: "Scientific fact checking corpus (BEIR format). Contains claims with corresponding abstracts.",
      stats: { docs: "5,183", queries: "300" },
      status: "Available"
    },
    {
      id: "hotpotqa",
      title: "HotpotQA",
      description: "Multi-hop question answering dataset.",
      stats: { docs: "Large", queries: "7,405" },
      status: "Available (Distractor)"
    },
    {
      id: "covidqa",
      title: "COVID-QA (RAGBench)",
      description: "Question answering about COVID-19 related scientific articles.",
      stats: { docs: "Varies", queries: "Varies" },
      status: "Partial Integration"
    }
  ]

  return (
    <div className="container mx-auto py-10 space-y-8">
      <div className="flex flex-col gap-2">
        <h1 className="text-4xl font-bold">Benchmark Datasets</h1>
        <p className="text-muted-foreground">Standardized datasets for text retrieval and generation evaluation.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {benchmarks.map((b) => (
          <Card key={b.id} className="flex flex-col p-6">
            <div className="mb-4">
              <div className="flex justify-between items-start">
                <h2 className="text-xl font-semibold">{b.title}</h2>
                <Badge variant={b.status.includes("Available") ? "brand" : "neutral"}>
                  {b.status}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-2">{b.description}</p>
            </div>
            <div className="mt-auto space-y-2 text-sm pt-4 border-t">
              <div className="flex justify-between pb-1">
                <span className="text-muted-foreground">Documents</span>
                <span className="font-medium">{b.stats.docs}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Queries</span>
                <span className="font-medium">{b.stats.queries}</span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
