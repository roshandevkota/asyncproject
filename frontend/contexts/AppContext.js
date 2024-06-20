import React, { createContext, useState, useContext, useEffect } from 'react';
import Cookies from 'js-cookie';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [state, setState] = useState({
    fileId: Cookies.get('fileId') || null,
    profilePath: Cookies.get('profilePath') || '',
    targetColumn: Cookies.get('targetColumn') || '',
    loadingUpload: Cookies.get('loadingUpload') === 'true',
    loadingProfile: Cookies.get('loadingProfile') === 'true',
    loadingTrain: Cookies.get('loadingTrain') === 'true',
    loadingPredict: Cookies.get('loadingPredict') === 'true',
    fileName: Cookies.get('fileName') || '',
    detectedDelimiter: Cookies.get('detectedDelimiter') || null,
    selectedDelimiter: Cookies.get('selectedDelimiter') || null,
    numTrials: parseInt(Cookies.get('numTrials') || '10', 10)  // Default to 10 if not provided
  });

  useEffect(() => {
    Cookies.set('fileId', state.fileId, { expires: 1 });
    Cookies.set('profilePath', state.profilePath, { expires: 1 });
    Cookies.set('targetColumn', state.targetColumn, { expires: 1 });
    Cookies.set('loadingUpload', state.loadingUpload.toString(), { expires: 1 });
    Cookies.set('loadingProfile', state.loadingProfile.toString(), { expires: 1 });
    Cookies.set('loadingTrain', state.loadingTrain.toString(), { expires: 1 });
    Cookies.set('loadingPredict', state.loadingPredict.toString(), { expires: 1 });
    Cookies.set('fileName', state.fileName, { expires: 1 });
    Cookies.set('detectedDelimiter', state.detectedDelimiter, { expires: 1 });
    Cookies.set('selectedDelimiter', state.selectedDelimiter, { expires: 1 });
    Cookies.set('numTrials', state.numTrials, { expires: 1 });
  }, [state]);

  const clearState = () => {
    setState({
      fileId: null,
      profilePath: '',
      targetColumn: '',
      loadingUpload: false,
      loadingProfile: false,
      loadingTrain: false,
      loadingPredict: false,
      fileName: '',
      detectedDelimiter: null,
      selectedDelimiter: null,
      numTrials: 10  // Reset to default value
    });
    Cookies.remove('fileId');
    Cookies.remove('profilePath');
    Cookies.remove('targetColumn');
    Cookies.remove('loadingUpload');
    Cookies.remove('loadingProfile');
    Cookies.remove('loadingTrain');
    Cookies.remove('loadingPredict');
    Cookies.remove('fileName');
    Cookies.remove('detectedDelimiter');
    Cookies.remove('selectedDelimiter');
    Cookies.remove('numTrials');
  };

  return (
    <AppContext.Provider value={{ state, setState, clearState }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => {
  return useContext(AppContext);
};
