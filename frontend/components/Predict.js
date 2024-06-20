import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAppContext } from '../contexts/AppContext';
import { PREDICT_URL, COLUMN_LIST_URL } from '../utils/constants';
import { Button, Spinner, Container, Form, Table, Tabs, Tab } from 'react-bootstrap';

const Predict = () => {
  const { state } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [columns, setColumns] = useState([]);
  const [singleData, setSingleData] = useState({});
  const [file, setFile] = useState(null);
  const [predictionHtmlSingle, setPredictionHtmlSingle] = useState('');
  const [predictionHtmlGroup, setPredictionHtmlGroup] = useState('');

  const fetchColumns = async () => {
    try {
      const response = await axios.post(COLUMN_LIST_URL, { file_id: state.fileId });
      setColumns(response.data.columns);
    } catch (error) {
      console.error('Error fetching columns:', error);
    }
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSingleDataChange = (column, value) => {
    setSingleData((prevData) => ({
      ...prevData,
      [column]: value,
    }));
  };

  const handleGroupPredict = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_id', state.fileId);
    formData.append('prediction_type', 'group');

    try {
      const response = await axios.post(PREDICT_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPredictionHtmlGroup(response.data.prediction_html);
    } catch (error) {
      console.error('Error predicting group data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSinglePredict = async () => {
    setLoading(true);

    try {
      const response = await axios.post(PREDICT_URL, {
        file_id: state.fileId,
        single_data: JSON.stringify(singleData),
        prediction_type: 'single'
      });
      setPredictionHtmlSingle(response.data.prediction_html);
    } catch (error) {
      console.error('Error predicting single data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchColumns();
  }, []);

  return (
    <Container className="mt-5">
      <h2 className="mb-4">Predict</h2>
      <Tabs defaultActiveKey="group" id="predict-tabs" className="mb-3">
        <Tab eventKey="group" title="Group">
          <Form.Group controlId="formFile" className="mb-3">
            <Form.Label>Choose CSV File</Form.Label>
            <Form.Control type="file" onChange={handleFileChange} />
          </Form.Group>
          <Button onClick={handleGroupPredict} variant="primary" disabled={loading || !file}>
            {loading ? (
              <>
                <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                {' '}Loading...
              </>
            ) : 'Predict Group Data'}
          </Button>
          {predictionHtmlGroup && (
            <div className="mt-3">
              <h3>Group Prediction Results</h3>
              <div dangerouslySetInnerHTML={{ __html: predictionHtmlGroup }} />
            </div>
          )}
        </Tab>
        <Tab eventKey="single" title="Single">
          {columns.length > 0 && (
            <>
              <Form className="mb-3">
                {columns.filter(column => column !== state.targetColumn).map(column => (
                  <Form.Group controlId={`form${column}`} key={column} className="mb-3">
                    <Form.Label>{column}</Form.Label>
                    <Form.Control
                      type="text"
                      onChange={(e) => handleSingleDataChange(column, e.target.value)}
                    />
                  </Form.Group>
                ))}
              </Form>
              <Button onClick={handleSinglePredict} variant="primary" disabled={loading}>
                {loading ? (
                  <>
                    <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                    {' '}Loading...
                  </>
                ) : 'Predict Single Data'}
              </Button>
              {predictionHtmlSingle && (
                <div className="mt-3">
                  <h3>Single Prediction Results</h3>
                  <div dangerouslySetInnerHTML={{ __html: predictionHtmlSingle }} />
                </div>
              )}
            </>
          )}
        </Tab>
      </Tabs>
    </Container>
  );
};

export default Predict;
