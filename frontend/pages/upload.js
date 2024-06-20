import React from 'react';
import FileUpload from '../components/FileUpload';
import useIsClient from '../hooks/useIsClient';
import { Container } from 'react-bootstrap';

const UploadPage = () => {
  const isClient = useIsClient();

  if (!isClient) {
    return null; // Render nothing on the server side
  }

  return (
    <Container className="mt-5">
      <FileUpload />
    </Container>
  );
};

export default UploadPage;
