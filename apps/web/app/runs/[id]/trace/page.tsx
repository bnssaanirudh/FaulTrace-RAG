import TracePage from "./trace-page";

export const metadata = {
  title: "Run Trace & Attribution - FaultTrace RAG",
};

export default function Page({ params }: { params: { id: string } }) {
  return <TracePage runId={params.id} />;
}
