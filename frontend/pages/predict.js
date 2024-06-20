import React from 'react';
import Predict from '../components/Predict';
import useIsClient from '../hooks/useIsClient';
import { Container } from 'react-bootstrap';

const PredictPage = () => {
  const isClient = useIsClient();

  if (!isClient) {
    return null; // Render nothing on the server side
  }

  return (
    <Container className="mt-5">
      <Predict />
    </Container>
  );
};

export default PredictPage;
