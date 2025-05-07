<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Patent Analysis Platform - README</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f8f8f8;
      margin: 0;
      padding: 2em;
      color: #333;
    }
    a {
      color: #0056b3;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    img {
      max-width: 100%;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      padding: 2em;
      box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);
    }
    .gallery {
      display: flex;
      flex-wrap: wrap;
      gap: 1em;
    }
    .gallery div {
      flex: 1 1 48%;
      text-align: center;
    }
    iframe {
      border: none;
    }
  </style>
</head>
<body>

<div class="container">
  <h1 style="text-align: center;">Patent Analysis Platform</h1>
  <p style="font-size: 1.2em;">The webpage is published at:</p>
  <p><a href="https://patent-analysis-e57ee02125bf.herokuapp.com/" target="_blank">
    https://patent-analysis-e57ee02125bf.herokuapp.com/</a></p>

  <div style="display: flex; align-items: flex-start; margin-top: 2em;">
    <img src="images/dfp.jpg" alt="Patent Analysis Graphic"
         style="width: 140px; margin-right: 1.5em; border: 1px solid #ccc; border-radius: 6px;">
    <div>
      <h2>Welcome to the Patent Analysis Platform</h2>
      <p>
        This web platform provides data analysis tools tailored for intellectual property professionals.
        You can generate insights and visualizations from international patent data in just a few steps.
      </p>
    </div>
  </div>

  <h2 style="margin-top: 2em;">How to Use</h2>
  <ol>
    <li>Go to the <a href="https://patentscope.wipo.int/search/en/search.jsf" target="_blank">WIPO Patentscope</a> search page.</li>
    <li>Sign in to your WIPO account to enable downloading results.</li>
    <li>Perform your patent search using any filters of interest (IPC codes, keywords, applicant, etc.).</li>
    <li>Download the result list as an Excel file (XLS format).</li>
    <li>Return to the platform and upload the Excel file to generate visual analytics.</li>
  </ol>

  <h3 style="margin-top: 2em;">Watch this short tutorial:</h3>
  <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border: 1px solid #ccc; border-radius: 8px;">
    <iframe src="https://www.youtube.com/embed/eo31FWVMW_o?si=hmpVeIAeSrjxdBOC"
            allowfullscreen
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
    </iframe>
  </div>

  <p style="margin-top: 2em;">
    This website is under development. For now, login is not required to access advanced features.
    Please be patient with the generation time — the platform uses limited cloud resources.
    Avoid uploading datasets with more than 3000 rows to prevent timeouts or failures.
    In future updates, signing up will allow users to save their searches and generated visuals.
  </p>

  <hr style="margin: 3em 0;">

  <h2>Gallery of Sample Visualizations</h2>
  <p>Explore example graphics generated from international patent data.</p>

  <h3>Countries Statistics</h3>
  <div class="gallery">
    <div>
      <img src="images/frequency_priority_years_r.png" alt="Priority Timeline">
      <p>Priority patent filing, yearly frequency</p>
    </div>
    <div>
      <img src="images/top_priority_countries_r.png" alt="Top Priority Countries">
      <p>Top Priority Filing Countries</p>
    </div>
    <div>
      <img src="images/top_countries_r.png" alt="Top Countries">
      <p>Top 10 countries by publication</p>
    </div>
    <div>
      <img src="images/origin_destcountr_r.png" alt="Origin-Destination Map">
      <p>Patent priority vs destination countries</p>
    </div>
  </div>

  <h3 style="margin-top: 2em;">Word Frequency Statistics: Wordclouds</h3>
  <div class="gallery">
    <div>
      <img src="images/wcld_nouns_r.png" alt="Nouns Wordcloud">
      <p>Wordcloud (Nouns)</p>
    </div>
    <div>
      <img src="images/wcld_verbs_r.png" alt="Verbs Wordcloud">
      <p>Wordcloud (Verbs)</p>
    </div>
    <div>
      <img src="images/wcld_adjectives_r.png" alt="Adjectives Wordcloud" style="width: 70%;">
      <p>Wordcloud (Adjectives)</p>
    </div>
  </div>

  <h3 style="margin-top: 2em;">IPC and Applicants Statistics</h3>
  <div class="gallery">
    <div>
      <img src="images/top_ipcs_r.png" alt="Top IPC Codes">
      <p>Top 10 IPC groups</p>
    </div>
    <div>
      <img src="images/parallel_coordinates_r.png" alt="Tech Evolution">
      <p>Technology evolution over time</p>
    </div>
    <div>
      <img src="images/Top20Appl_r.png" alt="Top 20 Applicants">
      <p>Top 20 Applicants</p>
    </div>
    <div>
      <img src="images/Applicants_parallel_r.png" alt="Top 5 Applicants Timeline">
      <p>Top 5 Applicants patent publication timeline</p>
    </div>
    <div>
      <img src="images/TopAppl_IPC_r.png" alt="IPC by Applicant">
      <p>Most frequent IPC groups by Applicant</p>
    </div>
  </div>

  <h3 style="margin-top: 2em;">Technology Transfer Landscape</h3>
  <div class="gallery">
    <div>
      <img src="images/network_plot.png" alt="Tech Transfer Network">
      <p>Flow of Technological Know-How Through Shared Inventors</p>
      <p>Applicants in the graph may have further connections out of this circle</p>
    </div>
  </div>
</div>

</body>
</html>
