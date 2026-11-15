import React from 'react';
import ReactDOM from 'react-dom/client';
import { MantineProvider } from '@mantine/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@mantine/core/styles.css';
import './styles.css';
import { Workbench } from './workbench';

const client = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: true } } });
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><MantineProvider defaultColorScheme="light">
    <QueryClientProvider client={client}><Workbench /></QueryClientProvider>
  </MantineProvider></React.StrictMode>,
);
