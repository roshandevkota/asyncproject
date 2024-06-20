import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAppContext } from '../contexts/AppContext';
import { GET_METADATA_URL, UPDATE_METADATA_URL, COLUMN_LIST_URL, TRAIN_MODEL_URL } from '../utils/constants';
import { Button, Spinner, Container, Table, Form, Row, Col, Card, Alert } from 'react-bootstrap';

const Train = () => {
  const { state, setState } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState(state.targetColumn || '');
  const [defaultMetadata, setDefaultMetadata] = useState({});
  const [modifiedMetadata, setModifiedMetadata] = useState({});
  const [isEditing, setIsEditing] = useState(false);
  const [editedMetadata, setEditedMetadata] = useState({});
  const [trainingInfo, setTrainingInfo] = useState(null);

  const fetchColumns = async () => {
    try {
      const response = await axios.post(COLUMN_LIST_URL, { file_id: state.fileId });
      setColumns(response.data.columns);
    } catch (error) {
      console.error('Error fetching columns:', error);
    }
  };

  const fetchMetadata = async () => {
    setLoading(true);
    try {
      const response = await axios.post(GET_METADATA_URL, { file_id: state.fileId, target_column: targetColumn });
      setDefaultMetadata(response.data.default_metadata);
      setModifiedMetadata(response.data.modified_metadata);
      setTrainingInfo(response.data.training_info);
      setState(prevState => ({
        ...prevState,
        defaultMetadata: response.data.default_metadata,
        modifiedMetadata: response.data.modified_metadata,
        targetColumn: targetColumn,
        numTrials: response.data.training_info ? response.data.training_info.num_trials : prevState.numTrials
      }));
    } catch (error) {
      console.error('Error getting metadata:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateMetadata = async () => {
    try {
      const response = await axios.post(UPDATE_METADATA_URL, {
        file_id: state.fileId,
        modified_metadata: editedMetadata
      });
      setModifiedMetadata(response.data.modified_metadata);
      setState(prevState => ({
        ...prevState,
        modifiedMetadata: response.data.modified_metadata
      }));
      setIsEditing(false);
    } catch (error) {
      console.error('Error updating metadata:', error);
    }
  };

  const handleEditClick = () => {
    setEditedMetadata(modifiedMetadata);
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedMetadata({});
  };

  const handleMetadataChange = (column, isCat) => {
    setEditedMetadata(prevMetadata => ({
      ...prevMetadata,
      [column]: {
        ...prevMetadata[column],
        is_cat: isCat === 'cat'
      }
    }));
  };

  const handleTrainModel = async () => {
    setLoading(true);
    try {
      const response = await axios.post(TRAIN_MODEL_URL, {
        file_id: state.fileId,
        target_column: targetColumn,
        num_trials: state.numTrials
      });
      setTrainingInfo(response.data.training_info);
      setState(prevState => ({
        ...prevState,
        loadingTrain: false
      }));
    } catch (error) {
      console.error('Error training model:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchColumns();
    if (targetColumn) {
      fetchMetadata();
    }
  }, []);

  useEffect(() => {
    if (state.defaultMetadata && state.modifiedMetadata) {
      setDefaultMetadata(state.defaultMetadata);
      setModifiedMetadata(state.modifiedMetadata);
      setTargetColumn(state.targetColumn);
    }
  }, [state.defaultMetadata, state.modifiedMetadata, state.targetColumn]);

  return (
    <Container className="mt-5">
      <h2 className="mb-4">Train Model</h2>
      <Form.Group controlId="targetColumn">
        <Form.Label>Target Column</Form.Label>
        <Form.Control
          as="select"
          value={targetColumn}
          onChange={e => setTargetColumn(e.target.value)}
        >
          <option value="">Select target column</option>
          {columns.map(column => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </Form.Control>
      </Form.Group>
      <Button onClick={fetchMetadata} variant="primary" disabled={loading || !targetColumn} className="mt-3 mb-4">
        {loading ? (
          <>
            <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
            {' '}Loading...
          </>
        ) : 'Get Metadata'}
      </Button>
      {Object.keys(defaultMetadata).length > 0 && (
        <>
          <Row className="mt-3">
            <Col md={6}>
              <Card>
                <Card.Header><h3>Default Metadata</h3></Card.Header>
                <Card.Body>
                  <Table striped bordered hover>
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(defaultMetadata).map(column => (
                        <tr key={column}>
                          <td>{column}</td>
                          <td>{defaultMetadata[column].is_cat ? 'Categorical' : 'Non-categorical'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </Card.Body>
              </Card>
            </Col>
            <Col md={6}>
              <Card>
                <Card.Header><h3>Modified Metadata</h3></Card.Header>
                <Card.Body>
                  <Table striped bordered hover>
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(modifiedMetadata).map(column => (
                        <tr key={column}>
                          <td>{column}</td>
                          <td>{modifiedMetadata[column].is_cat ? 'Categorical' : 'Non-categorical'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                  <Button onClick={handleEditClick} variant="info" className="mt-3">
                    Edit Metadata
                  </Button>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          {isEditing && (
            <Card className="mt-3">
              <Card.Header><h3>Edit Metadata</h3></Card.Header>
              <Card.Body>
                <Table striped bordered hover>
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(editedMetadata).map(column => (
                      <tr key={column}>
                        <td>{column}</td>
                        <td>
                          <Form.Control
                            as="select"
                            value={editedMetadata[column].is_cat ? 'cat' : 'non_cat'}
                            onChange={e => handleMetadataChange(column, e.target.value)}
                          >
                            <option value="cat">Categorical</option>
                            <option value="non_cat">Non-categorical</option>
                          </Form.Control>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
                <div className="d-flex justify-content-end">
                  <Button onClick={handleUpdateMetadata} variant="success" className="me-2">
                    Apply Changes
                  </Button>
                  <Button onClick={handleCancelEdit} variant="secondary">
                    Cancel
                  </Button>
                </div>
              </Card.Body>
            </Card>
          )}
          <Form.Group controlId="numTrials" className="mt-3">
            <Form.Label>Number of Trials</Form.Label>
            <Form.Control
              type="number"
              value={state.numTrials}
              onChange={e => setState({ ...state, numTrials: e.target.value })}
              min="1"
              disabled={loading}
            />
          </Form.Group>
          <Button onClick={handleTrainModel} variant="primary" className="mt-3 mb-4" disabled={loading}>
            {loading ? (
              <>
                <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                {' '}Training...
              </>
            ) : 'Train Model'}
          </Button>
          {trainingInfo && (
            <Alert variant="info" className="mt-3">
              <p>Status: {trainingInfo.status}</p>
              <p>Target Type: {trainingInfo.target_type}</p>
              <p>Model Type: {trainingInfo.model_type}</p>
              <p>Performance Score: {trainingInfo.performance_score}</p>
              <p>Number of Trials: {trainingInfo.num_trials}</p>
            </Alert>
          )}
        </>
      )}
    </Container>
  );
};

export default Train;

