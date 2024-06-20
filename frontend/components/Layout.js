import React from 'react';
import { Container, Navbar, Nav } from 'react-bootstrap';
import Link from 'next/link';

const Layout = ({ children }) => {
  return (
    <div>
      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Link href="/" passHref legacyBehavior>
            <Navbar.Brand>Async Django</Navbar.Brand>
          </Link>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Link href="/upload" passHref legacyBehavior>
                <Nav.Link>Upload</Nav.Link>
              </Link>
              <Link href="/profile" passHref legacyBehavior>
                <Nav.Link>Profile</Nav.Link>
              </Link>
              <Link href="/train" passHref legacyBehavior>
                <Nav.Link>Train</Nav.Link>
              </Link>
              <Link href="/predict" passHref legacyBehavior>
                <Nav.Link>Predict</Nav.Link>
              </Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      <Container className="mt-4">
        {children}
      </Container>
    </div>
  );
};

export default Layout;
