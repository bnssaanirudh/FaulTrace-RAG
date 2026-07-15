import { DashboardPage } from '@/app/(dashboard)/dashboard-page';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Dashboard — FaultTrace-RAG',
  description: 'Overview of corpus worlds, query coverage, and pipeline runs.',
};

export default function Page() {
  return <DashboardPage />;
}
