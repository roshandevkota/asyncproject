import React from 'react';
import useIsClient from '../hooks/useIsClient';
import { Container } from 'react-bootstrap';

const HomePage = () => {
  const isClient = useIsClient();

  if (!isClient) {
    return null; // Render nothing on the server side
  }

  return (
    <Container className="mt-5">
      <h1>Welcome to the Async ML App</h1>
      <p>Use the sidebar to navigate between different pages.</p>
    </Container>
  );
};

export default HomePage;
