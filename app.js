// App State
let appState = {
  rfpFile: null,
  proposalFile: null,
  analysisRun: false,
  analysisResults: null,
  filteredRequirements: []
};

// DOM Elements
const navLinks = document.querySelectorAll('.nav-link');
const pageSections = document.querySelectorAll('.page-section');
const startAnalysisBtn = document.getElementById('start-analysis-btn');

const rfpUploadCard = document.getElementById('rfp-upload-card');
const rfpFileInput = document.getElementById('rfp-file-input');
const rfpFileInfo = document.getElementById('rfp-file-info');

const proposalUploadCard = document.getElementById('proposal-upload-card');
const proposalFileInput = document.getElementById('proposal-file-input');
const proposalFileInfo = document.getElementById('proposal-file-info');

const analyzeProposalBtn = document.getElementById('analyze-proposal-btn');
const analysisLoader = document.getElementById('analysis-loader');
const loaderStatusText = document.getElementById('loader-status-text');
const loaderPercentage = document.getElementById('loader-percentage');
const loaderProgressFill = document.getElementById('loader-progress-fill');

const dashboardReport = document.getElementById('dashboard-report');

// Collapsible Sidebar & Re-upload DOM Elements
const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
const appContainer = document.querySelector('.app-container');
const uploadSectionFull = document.getElementById('upload-section-full');
const newAnalysisBtn = document.getElementById('new-analysis-btn');

// Mobile Responsive UI Elements
const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');

// Sidebar toggle logic with local storage persistence
if (sidebarToggleBtn && appContainer) {
  if (localStorage.getItem('sidebar-collapsed') === 'true') {
    appContainer.classList.add('sidebar-collapsed');
  }
  sidebarToggleBtn.addEventListener('click', () => {
    const isCollapsed = appContainer.classList.toggle('sidebar-collapsed');
    localStorage.setItem('sidebar-collapsed', isCollapsed);
  });
}

// New Analysis trigger button click handler
if (newAnalysisBtn) {
  newAnalysisBtn.addEventListener('click', () => {
    newAnalysisBtn.style.display = 'none';
    dashboardReport.style.display = 'none';
    uploadSectionFull.style.display = 'block';
    
    // Reset file fields
    rfpFileInput.value = '';
    proposalFileInput.value = '';
    rfpFileInfo.style.display = 'none';
    proposalFileInfo.style.display = 'none';
    rfpUploadCard.classList.remove('file-selected');
    proposalUploadCard.classList.remove('file-selected');
    
    appState.rfpFile = null;
    appState.proposalFile = null;
    appState.analysisRun = false;
    appState.analysisResults = null;
    
    analyzeProposalBtn.disabled = true;
    analyzeProposalBtn.style.opacity = '0.6';
    analyzeProposalBtn.style.cursor = 'not-allowed';
    
    resetLoader();
    uploadSectionFull.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

// Report Tab Switching Navigation Logic
const reportTabButtons = document.querySelectorAll('.report-tab-btn');
const reportTabContents = document.querySelectorAll('.report-tab-content');

if (reportTabButtons && reportTabContents) {
  reportTabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.dataset.reportTab;
      
      // Update active button state
      reportTabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Update active content state
      reportTabContents.forEach(content => {
        if (content.id === targetTab) {
          content.classList.add('active');
        } else {
          content.classList.remove('active');
        }
      });
    });
  });
}

// Mobile hamburger menu toggle click handler
if (mobileMenuBtn && appContainer) {
  mobileMenuBtn.addEventListener('click', () => {
    appContainer.classList.add('sidebar-open');
  });
}

// Mobile backdrop click to close sidebar drawer
if (sidebarBackdrop && appContainer) {
  sidebarBackdrop.addEventListener('click', () => {
    appContainer.classList.remove('sidebar-open');
  });
}

// Table Explorer Elements
const requirementsTbody = document.getElementById('requirements-tbody');
const explorerEmptyState = document.getElementById('explorer-empty-state');
const reqSearchInput = document.getElementById('req-search-input');
const filterCategory = document.getElementById('filter-category');
const filterStatus = document.getElementById('filter-status');
const exportResultsBtn = document.getElementById('export-results-btn');

/* ================================== ROUTING (SPA) ================================== */
function switchPage(pageId) {
  // Close mobile sidebar drawer if it was open
  if (appContainer) {
    appContainer.classList.remove('sidebar-open');
  }

  // Update nav links active states
  navLinks.forEach(link => {
    if (link.dataset.page === pageId) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Update page sections display
  pageSections.forEach(section => {
    if (section.id === pageId) {
      section.classList.add('active');
    } else {
      section.classList.remove('active');
    }
  });

  // Scroll main view to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Bind navigation clicks
navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    switchPage(link.dataset.page);
  });
});

// Welcome Page Start Analysis CTA Button
startAnalysisBtn.addEventListener('click', () => {
  switchPage('dashboard-page');
});

// Inline links binding
document.addEventListener('click', (e) => {
  if (e.target && e.target.dataset.link) {
    e.preventDefault();
    switchPage(e.target.dataset.link);
  }
});


/* ================================== FILE UPLOAD LOGIC ================================== */
function setupUploadCard(card, input, infoContainer, fileKey) {
  // Trigger click on file input
  card.addEventListener('click', (e) => {
    if (e.target.tagName !== 'INPUT' && !e.target.closest('.file-info')) {
      input.click();
    }
  });

  // Input change
  input.addEventListener('change', (e) => {
    if (input.files.length > 0) {
      handleFileSelected(input.files[0], card, infoContainer, fileKey);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    card.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      card.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    card.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      card.classList.remove('dragover');
    }, false);
  });

  card.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0 && files[0].type === 'application/pdf') {
      handleFileSelected(files[0], card, infoContainer, fileKey);
    } else {
      alert('Please upload a valid PDF file.');
    }
  });
}

function handleFileSelected(file, card, infoContainer, fileKey) {
  card.classList.add('file-selected');
  
  // Show file info badge
  infoContainer.querySelector('.file-name').textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  infoContainer.style.display = 'inline-flex';
  
  // Update state
  appState[fileKey] = file;
  
  // Enable analyze button if both are uploaded
  if (appState.rfpFile && appState.proposalFile) {
    analyzeProposalBtn.disabled = false;
    analyzeProposalBtn.style.opacity = '1';
    analyzeProposalBtn.style.cursor = 'pointer';
  }
}

setupUploadCard(rfpUploadCard, rfpFileInput, rfpFileInfo, 'rfpFile');
setupUploadCard(proposalUploadCard, proposalFileInput, proposalFileInfo, 'proposalFile');


/* ================================== API & LOADER INTEGRATION ================================== */
analyzeProposalBtn.addEventListener('click', () => {
  // Start loading UI sequence
  analyzeProposalBtn.style.display = 'none';
  analysisLoader.style.display = 'flex';
  
  let progress = 0;
  let isFetchDone = false;
  let fetchError = null;

  // Start real API request
  const formData = new FormData();
  formData.append('rfp_file', appState.rfpFile);
  formData.append('proposal_file', appState.proposalFile);

  fetch('/api/analyze', {
    method: 'POST',
    body: formData
  })
  .then(res => {
    if (!res.ok) {
      throw new Error("Backend document parsing error.");
    }
    return res.json();
  })
  .then(data => {
    appState.analysisResults = data;
    appState.filteredRequirements = [...data.requirements];
    isFetchDone = true;
  })
  .catch(err => {
    console.error(err);
    fetchError = err.message;
    isFetchDone = true;
  });

  // Progress animation runner
  function runProgress() {
    if (fetchError) {
      alert(`Error during analysis: ${fetchError}`);
      resetLoader();
      return;
    }

    if (progress < 95 || (progress < 100 && isFetchDone)) {
      progress += Math.floor(Math.random() * 6) + 3;
      
      // Hold at 95% if server hasn't responded yet
      if (progress > 95 && !isFetchDone) {
        progress = 95;
      }
      if (progress > 100) progress = 100;

      // Update loading status phrases
      let phrase = "Reading binary headers...";
      if (progress > 80) phrase = "Calculating Win Probability indexes...";
      else if (progress > 60) phrase = "Running NLP similarity matching...";
      else if (progress > 40) phrase = "Generating compliance matrix profiles...";
      else if (progress > 20) phrase = "Parsing layout structure & extracting texts...";

      loaderStatusText.textContent = phrase;
      loaderProgressFill.style.width = `${progress}%`;
      loaderPercentage.textContent = `${progress}%`;

      if (progress < 100) {
        setTimeout(runProgress, 140);
      } else {
        finishAnalysis(appState.analysisResults);
      }
    } else if (progress === 95 && !isFetchDone) {
      // Hold at 95% and poll for fetch completion
      setTimeout(runProgress, 200);
    }
  }

  runProgress();
});

function resetLoader() {
  analysisLoader.style.display = 'none';
  analyzeProposalBtn.style.display = 'inline-flex';
  analyzeProposalBtn.disabled = false;
}

function finishAnalysis(data) {
  if (!data) return;
  appState.analysisRun = true;
  
  // Update KPI displays safely
  const compScoreEl = document.getElementById('kpi-compliance-score');
  if (compScoreEl) compScoreEl.textContent = data.kpis.complianceScore;
  
  const winProbEl = document.getElementById('kpi-win-probability');
  if (winProbEl) winProbEl.textContent = data.kpis.winProbability;
  
  const strongMatchesEl = document.getElementById('kpi-strong-matches');
  if (strongMatchesEl) strongMatchesEl.textContent = data.kpis.strongMatches;
  
  const missingReqsEl = document.getElementById('kpi-missing-reqs');
  if (missingReqsEl) missingReqsEl.textContent = data.kpis.missingRequirements;
  
  const avgSimilarityEl = document.getElementById('kpi-avg-similarity');
  if (avgSimilarityEl) avgSimilarityEl.textContent = data.kpis.avgSimilarity;

  // Show Report panel
  dashboardReport.style.display = 'block';

  // Animate Coverage Progress bars dynamically
  Object.entries(data.coverage).forEach(([key, val]) => {
    const el = document.getElementById(`cov-${key}-fill`);
    const textEl = document.getElementById(`cov-${key}-val`);
    if (el) {
      el.style.width = '0%';
      let count = 0;
      const interval = setInterval(() => {
        if (count < val) {
          count++;
          textEl.textContent = `${count}%`;
        } else {
          clearInterval(interval);
        }
      }, 10);
      el.style.width = `${val}%`;
    }
  });

  // Render Risk summaries dynamically
  const riskList = document.querySelector('.risk-list');
  riskList.innerHTML = '';
  data.risks.forEach(risk => {
    const riskClass = risk.status === 'high' ? 'high-risk' : 'med-risk';
    const badgeSymbol = risk.status === 'high' ? '❌' : '⚠️';
    const riskDiv = document.createElement('div');
    riskDiv.className = `risk-item ${riskClass}`;
    riskDiv.innerHTML = `
      <div class="risk-badge">${badgeSymbol}</div>
      <div class="risk-info">
        <div class="risk-name">${risk.category}</div>
        <div class="risk-desc">${risk.desc}</div>
      </div>
    `;
    riskList.appendChild(riskDiv);
  });

  // Render AI Expected Win Improvement
  const winValEl = document.getElementById('win-improvement-val');
  if (winValEl) {
    // Keep it dynamic, but format to show a range if appropriate, or use the dynamic value directly
    winValEl.textContent = data.insights.winImprovement || '+5% – 10%';
  }
  
  // Populate Strengths list in Assessment card
  const strengthsUl = document.getElementById('assessment-strengths-list');
  if (strengthsUl) {
    strengthsUl.innerHTML = '';
    data.insights.strengths.forEach(str => {
      const li = document.createElement('li');
      li.className = 'insight-bullet';
      li.textContent = str;
      strengthsUl.appendChild(li);
    });
  }

  // Populate Areas for Improvement in Assessment card
  const improvementsUl = document.getElementById('assessment-improvements-list');
  if (improvementsUl) {
    improvementsUl.innerHTML = '';
    data.insights.weaknesses.forEach(wk => {
      const li = document.createElement('li');
      li.className = 'insight-bullet';
      li.textContent = wk;
      improvementsUl.appendChild(li);
    });
  }

  // Helper function to generate clean consulting report titles from tags
  function getRecommendationTitle(rec) {
    if (rec.title) return rec.title;
    const tag = (rec.tag || '').toLowerCase().trim();
    if (tag.includes('security')) {
      return 'Strengthen Security Controls';
    } else if (tag.includes('integration')) {
      return 'Enhance Integration Architecture';
    } else if (tag.includes('performance') || tag.includes('technical')) {
      return 'Optimize System Performance';
    } else if (tag.includes('timeline') || tag.includes('schedule') || tag.includes('warranty')) {
      return 'Align Project Timeline and SLA';
    } else if (tag.includes('budget') || tag.includes('cost') || tag.includes('financial')) {
      return 'Improve Financial Proposal Alignment';
    } else if (tag.includes('team') || tag.includes('experience') || tag.includes('vendor')) {
      return 'Strengthen Team & Vendor Qualifications';
    } else if (tag.includes('functional')) {
      return 'Address Functional Requirements';
    } else {
      // Capitalize first letter
      const tagStr = rec.tag || 'Proposal';
      return `Address ${tagStr.charAt(0).toUpperCase() + tagStr.slice(1)} Requirements`;
    }
  }

  // Render AI Numbered Recommendations (Advisory style)
  const recommendationsList = document.getElementById('recommendations-numbered-list');
  if (recommendationsList) {
    recommendationsList.innerHTML = '';
    data.insights.recommendations.forEach(rec => {
      const li = document.createElement('li');
      li.className = 'recommendation-report-item';
      
      const title = getRecommendationTitle(rec);
      
      li.innerHTML = `
        <div class="rec-item-title">${title}</div>
        <div class="rec-item-desc">${rec.text}</div>
      `;
      recommendationsList.appendChild(li);
    });
  }

  // Render Table Explorer contents
  renderRequirementsTable();
  
  // Toggle upload view to show header new analysis btn and hide full upload panel
  if (uploadSectionFull) {
    uploadSectionFull.style.display = 'none';
  }
  if (newAnalysisBtn) {
    newAnalysisBtn.style.display = 'inline-flex';
  }

  // Reset default active tab back to Metrics Dashboard on new analysis load
  const firstTabBtn = document.querySelector('.report-tab-btn[data-report-tab="tab-dashboard"]');
  if (firstTabBtn) {
    firstTabBtn.click();
  }

  // Smooth scroll main view to top so they see the Executive Summary immediately
  setTimeout(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, 100);
}


/* ================================== REQUIREMENTS EXPLORER ================================== */
function getStatusClass(status) {
  switch (status) {
    case 'strong': return 'strong';
    case 'partial': return 'partial';
    case 'missing': return 'missing';
    default: return '';
  }
}

function renderRequirementsTable() {
  requirementsTbody.innerHTML = '';
  
  if (!appState.analysisRun || !appState.analysisResults) {
    explorerEmptyState.style.display = 'flex';
    return;
  }
  
  explorerEmptyState.style.display = 'none';
  
  if (appState.filteredRequirements.length === 0) {
    requirementsTbody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; padding: 3rem; color: var(--color-text-light);">
          No matching requirements found for current filters.
        </td>
      </tr>
    `;
    return;
  }

  appState.filteredRequirements.forEach(req => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="col-req">${req.requirement}</td>
      <td class="col-cat"><span class="category-badge">${req.category}</span></td>
      <td class="col-match">${req.match}</td>
      <td class="col-score">${req.score}</td>
      <td class="col-status">
        <span class="status-badge ${getStatusClass(req.status)}">
          ${req.status === 'strong' ? 'Strong Match' : req.status === 'partial' ? 'Partial Match' : 'Missing'}
        </span>
      </td>
    `;
    requirementsTbody.appendChild(tr);
  });
}

function applyFilters() {
  if (!appState.analysisResults) return;
  
  const query = reqSearchInput.value.toLowerCase().trim();
  const category = filterCategory.value;
  const status = filterStatus.value;

  appState.filteredRequirements = appState.analysisResults.requirements.filter(req => {
    const matchesSearch = 
      req.requirement.toLowerCase().includes(query) ||
      req.match.toLowerCase().includes(query) ||
      req.category.toLowerCase().includes(query);
      
    const matchesCategory = category === 'all' || req.category === category;
    const matchesStatus = status === 'all' || req.status === status;

    return matchesSearch && matchesCategory && matchesStatus;
  });

  renderRequirementsTable();
}

reqSearchInput.addEventListener('input', applyFilters);
filterCategory.addEventListener('change', applyFilters);
filterStatus.addEventListener('change', applyFilters);


/* ================================== EXPORT CSV LOGIC ================================== */
exportResultsBtn.addEventListener('click', () => {
  if (!appState.analysisRun || appState.filteredRequirements.length === 0) {
    alert("Please run proposal analysis to yield exportable results.");
    return;
  }
  
  const headers = ["ID", "Requirement", "Category", "Best Proposal Match", "Similarity Score", "Status"];
  const rows = appState.filteredRequirements.map(req => [
    req.id,
    `"${req.requirement.replace(/"/g, '""')}"`,
    `"${req.category}"`,
    `"${req.match.replace(/"/g, '""')}"`,
    `"${req.score}"`,
    `"${req.status.toUpperCase()}"`
  ]);
  
  const csvContent = [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `BidWise_Analysis_Report_${new Date().toISOString().split('T')[0]}.csv`);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// Initial Empty Render
renderRequirementsTable();
