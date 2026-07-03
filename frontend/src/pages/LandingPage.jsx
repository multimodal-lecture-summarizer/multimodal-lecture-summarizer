/**
 * LandingPage — Trang chủ.
 * Trang: UIs/1_landing.html → frontend/src/pages/LandingPage.jsx
 */
import Navbar from '../components/Navbar';
import LandingHero from '../components/LandingHero';
import FeatureCards from '../components/FeatureCards';

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <LandingHero />
      <FeatureCards />
    </>
  );
}
