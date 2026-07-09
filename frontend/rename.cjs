const fs = require('fs');
const path = require('path');

function replaceInFile(filePath, searchRegex, replaceText) {
  let content = fs.readFileSync(filePath, 'utf8');
  let newContent = content.replace(searchRegex, replaceText);
  if (content !== newContent) {
    fs.writeFileSync(filePath, newContent, 'utf8');
    console.log(`Updated ${filePath}`);
  }
}

const files = [
  'index.html',
  'src/pages/LandingPage.tsx',
  'src/pages/HistoryPage.tsx',
  'src/pages/AuthPage.tsx',
  'src/i18n/vi.json',
  'src/i18n/en.json',
  'src/components/Header.tsx'
];

files.forEach(file => {
  const p = path.join(__dirname, file);
  if (fs.existsSync(p)) {
    replaceInFile(p, /Lumina/g, 'PrismVideo');
  }
});
