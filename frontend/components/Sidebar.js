import React from 'react';
import Link from 'next/link';
import styles from '../styles/sidebar.module.css';

const Sidebar = () => {
  return (
    <div className={styles.sidebar}>
      <h2>Navigation</h2>
      <ul>
        <li><Link href="/upload">Upload</Link></li>
        <li><Link href="/profile">Profile</Link></li>
        <li><Link href="/train">Train</Link></li>
        <li><Link href="/predict">Predict</Link></li>
      </ul>
    </div>
  );
};

export default Sidebar;
