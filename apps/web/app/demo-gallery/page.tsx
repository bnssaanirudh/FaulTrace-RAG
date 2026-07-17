export default function DemoGalleryPage() {
  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Demo Gallery</h2>
      </div>
      <div className="text-muted-foreground">
        <p>This gallery showcases the pre-generated plots, benchmarking reports, and system validation graphs.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div className="rounded-xl border bg-card text-card-foreground shadow">
          <div className="p-6 flex flex-col items-center justify-center space-y-4">
            <h3 className="font-semibold leading-none tracking-tight">Accuracy vs. Scale</h3>
            <div className="h-40 bg-muted/50 rounded-lg w-full flex items-center justify-center border border-dashed">
              <span className="text-sm text-muted-foreground">[Plot Placeholder]</span>
            </div>
          </div>
        </div>

        <div className="rounded-xl border bg-card text-card-foreground shadow">
          <div className="p-6 flex flex-col items-center justify-center space-y-4">
            <h3 className="font-semibold leading-none tracking-tight">Latency-Accuracy Tradeoff</h3>
            <div className="h-40 bg-muted/50 rounded-lg w-full flex items-center justify-center border border-dashed">
              <span className="text-sm text-muted-foreground">[Plot Placeholder]</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
