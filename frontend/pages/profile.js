import React from 'react';
import Profile from '../components/Profile';
import useIsClient from '../hooks/useIsClient';
import { Container } from 'react-bootstrap';

const ProfilePage = () => {
  const isClient = useIsClient();

  if (!isClient) {
    return null; // Render nothing on the server side
  }

  return (
    <Container className="mt-5">
      <Profile />
    </Container>
  );
};

export default ProfilePage;
