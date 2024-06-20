import React from 'react';
import Train from '../components/Train';
import useIsClient from '../hooks/useIsClient';
import { Container } from 'react-bootstrap';

const TrainPage = () => {
  const isClient = useIsClient();

  if (!isClient) {
    return null; // Render nothing on the server side
  }

  return (
    <Container className="mt-5">
      <Train />
    </Container>
  );
};

export default TrainPage;
