import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAppContext } from '../contexts/AppContext';
import { UPLOAD_URL, API_BASE_URL } from '../utils/constants';
import { Button, Spinner, Container, Alert, Form, Table, Row, Col } from 'react-bootstrap';
import Cookies from 'js-cookie';

const delimiterOptions = [
  { value: ',', label: 'Comma (,)' },
  { value: '\t', label: 'Tab (\\t)' },
  { value: ';', label: 'Semicolon (;)' },
  { value: ' ', label: 'Space ( )' }
];

const FileUpload = () => {
  const { state, setState, clearState } = useAppContext();
  const [localLoading, setLocalLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [delimiter, setDelimiter] = useState(state.selectedDelimiter || '');
  const [detectedDelimiter, setDetectedDelimiter] = useState(state.detectedDelimiter || '');
  const [dataPreview, setDataPreview] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLocalLoading(true);
    setState(prevState => ({ ...prevState, loadingUpload: true }));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(UPLOAD_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { file_id, delimiter: detectedDelimiter } = response.data;

      setState(prevState => ({
        ...prevState,
        fileId: file_id,
        fileName: file.name,
        loadingUpload: false,
        detectedDelimiter: detectedDelimiter,
        selectedDelimiter: detectedDelimiter
      }));

      setDetectedDelimiter(detectedDelimiter);
      setDelimiter(detectedDelimiter);
      fetchDataPreview(file_id, detectedDelimiter);

      setLocalLoading(false);
    } catch (error) {
      console.error('Error uploading file:', error);
      setState(prevState => ({ ...prevState, loadingUpload: false }));
      setLocalLoading(false);
    }
  };

  const fetchDataPreview = async (fileId, delimiter) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/data_preview/`, { file_id: fileId, delimiter });
      setDataPreview(response.data.preview);
    } catch (error) {
      console.error('Error fetching data preview:', error);
    }
  };

  const handleDelimiterChange = (event) => {
    const newDelimiter = event.target.value;
    setDelimiter(newDelimiter);
    setState(prevState => ({
      ...prevState,
      selectedDelimiter: newDelimiter
    }));
    fetchDataPreview(state.fileId, newDelimiter);
  };

  const handleClear = () => {
    clearState();
    setFile(null);
    setLocalLoading(false);
    setDetectedDelimiter(null);
    setDelimiter(null);
    setDataPreview(null);
  };

  useEffect(() => {
    setLocalLoading(state.loadingUpload);
  }, [state.loadingUpload]);

  useEffect(() => {
    const savedLoading = Cookies.get('loadingUpload') === 'true';
    if (savedLoading !== state.loadingUpload) {
      setState(prevState => ({ ...prevState, loadingUpload: savedLoading }));
      setLocalLoading(savedLoading);
    }
  }, []); // Empty dependency array ensures this runs only once on mount

  useEffect(() => {
    if (state.fileId && state.selectedDelimiter) {
      fetchDataPreview(state.fileId, state.selectedDelimiter);
    }
  }, [state.fileId, state.selectedDelimiter]);

  return (
    <Container className="mt-5">
      <h2>Upload File</h2>
      <Form.Group controlId="formFile" className="mb-3">
        <Form.Label>Choose CSV File</Form.Label>
        <Form.Control type="file" onChange={handleFileChange} />
      </Form.Group>
      <Button onClick={handleUpload} variant="primary" disabled={localLoading}>
        {localLoading ? (
          <>
            <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
            {' '}Loading...
          </>
        ) : 'Upload'}
      </Button>
      {state.fileName && (
        <>
          <Alert variant="success" className="mt-3">
            <h3>File Uploaded</h3>
            <p>{state.fileName}</p>
            <Button onClick={handleClear} variant="danger" className="mt-3">
              Clear All
            </Button>
          </Alert>
          <Row>
            <Col md={6}>
              <Form.Group controlId="formDetectedDelimiter" className="mb-3">
                <Form.Label>Detected Delimiter</Form.Label>
                <Form.Control as="select" value={detectedDelimiter} disabled>
                  {delimiterOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </Form.Control>
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Group controlId="formSelectedDelimiter" className="mb-3">
                <Form.Label>Selected Delimiter</Form.Label>
                <Form.Control as="select" value={delimiter} onChange={handleDelimiterChange}>
                  {delimiterOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </Form.Control>
              </Form.Group>
            </Col>
          </Row>
          {dataPreview && (
            <Table striped bordered hover>
              <thead>
                <tr>
                {Object.keys(dataPreview[0]).map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataPreview.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, idx) => (
                      <td key={idx}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </>
      )}
    </Container>
  );
};

export default FileUpload;

