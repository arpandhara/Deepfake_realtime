const express = require('express');
const multer = require('multer');

const router = express.Router();

// Use memory storage for Multer
const storage = multer.memoryStorage();
const upload = multer({ storage });

router.post('/', upload.single('chunk'), async (req, res) => {
  try {
    const aiNgrokUrl = process.env.AI_NGROK_URL;

    if (!aiNgrokUrl) {
      console.error('AI_NGROK_URL is not defined in environment variables.');
      return res.status(500).json({ error: 'AI server configuration missing.' });
    }

    if (!req.file || !req.file.buffer) {
      return res.status(400).json({ error: 'No video chunk uploaded.' });
    }

    // Convert buffer to base64 string
    const base64Image = req.file.buffer.toString('base64');

    // Use native fetch to POST the JSON payload to AI server
    const aiResponse = await fetch(`${aiNgrokUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_b64: base64Image }),
    });

    if (!aiResponse.ok) {
      const errorText = await aiResponse.text();
      throw new Error(`AI server responded with status ${aiResponse.status}: ${errorText}`);
    }

    // Await the JSON response and return it to the client
    const data = await aiResponse.json();
    return res.json(data);
  } catch (error) {
    console.error('Error forwarding frame to AI server:', error);
    return res.status(500).json({ error: 'Failed to process frame.' });
  }
});

module.exports = router;
