import type { Metadata } from 'next';
import { WorldsPage } from './worlds-page';

export const metadata: Metadata = { title: 'Worlds' };
export default function Page() { return <WorldsPage />; }
