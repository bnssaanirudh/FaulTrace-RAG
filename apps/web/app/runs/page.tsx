import type { Metadata } from 'next';
import { RunsPage } from './runs-page';

export const metadata: Metadata = { title: 'Pipeline Runs' };
export default function Page() { return <RunsPage />; }
