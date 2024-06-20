import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useAppContext } from '../contexts/AppContext';
import { PROFILE_URL, API_MEDIA_BASE_URL } from '../utils/constants';
import useIsClient from '../hooks/useIsClient';
import { Button, Spinner, Container } from 'react-bootstrap';
import Cookies from 'js-cookie';

const Profile = () => {
  const { state, setState } = useAppContext();
  const isClient = useIsClient();
  const [localLoading, setLocalLoading] = useState(state.loadingProfile);

  const handleProfile = async () => {
    setLocalLoading(true);
    setState(prevState => ({ ...prevState, loadingProfile: true }));
    try {
      const response = await axios.post(PROFILE_URL, { file_id: state.fileId });
      setState(prevState => ({
        ...prevState,
        profilePath: response.data.profile_path,
        loadingProfile: false
      }));
      Cookies.set('profilePath', response.data.profile_path);
      setLocalLoading(false);
    } catch (error) {
      console.error('Error generating profile:', error);
      setState(prevState => ({ ...prevState, loadingProfile: false }));
      setLocalLoading(false);
    }
  };

  useEffect(() => {
    setLocalLoading(state.loadingProfile);
  }, [state.loadingProfile]);

  useEffect(() => {
    const savedLoading = Cookies.get('loadingProfile') === 'true';
    if (savedLoading !== state.loadingProfile) {
      setState(prevState => ({ ...prevState, loadingProfile: savedLoading }));
      setLocalLoading(savedLoading);
    }
  }, []); // Empty dependency array ensures this runs only once on mount

  useEffect(() => {
    const savedProfilePath = Cookies.get('profilePath');
    if (savedProfilePath) {
      setState(prevState => ({ ...prevState, profilePath: savedProfilePath }));
    }
  }, []); // Empty dependency array ensures this runs only once on mount

  if (!isClient) {
    return null; // Render nothing on the server side
  }

  return (
    <Container className="mt-5">
      <h2>Generate Profile</h2>
      <Button onClick={handleProfile} variant="primary" disabled={localLoading}>
        {localLoading ? (
          <>
            <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
            {' '}Loading...
          </>
        ) : 'Generate Profile'}
      </Button>
      {state.profilePath && !localLoading && (
        <div className="mt-3">
          <h3>Profile Report</h3>
          <iframe
            src={`${API_MEDIA_BASE_URL}/media/${state.profilePath}`}
            style={{ width: '100%', height: '800px', border: 'none' }}
            title="Profile Report"
          />
        </div>
      )}
    </Container>
  );
};

export default Profile;
