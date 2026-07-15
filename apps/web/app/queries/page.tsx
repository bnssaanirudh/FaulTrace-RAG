import type { Metadata } from 'next';
import { QueriesPage } from './queries-page';

export const metadata: Metadata = { title: 'Queries' };
export default function Page() { return <QueriesPage />; }
