import type { Metadata } from 'next';
import { DashboardPage } from './dashboard-page';

export const metadata: Metadata = {
  title: 'Dashboard — FaultTrace-RAG',
  description: 'Overview of corpus worlds, query coverage, and pipeline runs.',
};

export default function Page() {
  return <DashboardPage />;
}
