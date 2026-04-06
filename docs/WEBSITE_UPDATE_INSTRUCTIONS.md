# Website Update Instructions

## Quick Copy-Paste for Your Website

I've created all the content files you need to update your website. Here's where to find them and how to use them:

---

## 📁 Files Created (Ready to Copy)

### 1. Homepage Update
**File:** `docs/WEBSITE_HOMEPAGE_UPDATE.md`

**What to update on your homepage:**
- Add "statistical rigor" to your bio/tagline
- Update featured project section with new SignalForge metrics
- Add confidence intervals to project showcase
- Update keywords for SEO

### 2. Project Page (Full HTML)
**File:** `docs/SIGNALFORGE_PROJECT_PAGE.html`

**What this includes:**
- Complete HTML for SignalForge project page
- All statistical rigor sections
- Performance tables with confidence intervals
- Calibration analysis
- Feature engineering with learned weights
- Business impact metrics
- SEO meta tags
- Schema.org markup

### 3. Short Version (For Projects Grid)
**File:** `docs/WEBSITE_UPDATE_FINAL.md` (bottom section)

**What to copy:**
- Short description for projects listing page
- Key metrics summary
- Quick links

---

## 🎯 What to Update

### Homepage (christiangcallahan.tech)

**1. Update Your Tagline:**
```
Old: "Data Scientist • MS Candidate"
New: "Data Scientist • MS Candidate • Production ML with Statistical Rigor"
```

**2. Update Featured Project:**
```markdown
## 🏆 SignalForge - Production Churn Intelligence

**The Challenge:** Predict customer churn with statistical rigor

**Key Results:**
- ✅ 0.850 ± 0.013 AUC with 5-fold cross-validation
- ✅ Statistically significantly better (p=0.0074 vs Random Forest)
- ✅ $1.67M annual revenue at risk identified
- ✅ 1.21x - 1.81x expected ROI

**What Makes This Different:**
- Bootstrap confidence intervals on all metrics
- Statistical significance testing (not just "best model")
- Calibration analysis for probability accuracy
- Learned feature weights (not hard-coded)
```

**3. Add Statistical Rigor to Skills:**
```
### Analytical Skills
- Statistical Rigor: Cross-validation, Bootstrap CI, Significance Testing
- Feature Engineering: Data-driven optimization with regularization
- Model Evaluation: AUC, Calibration, Business Metrics
```

---

### Project Page (christiangcallahan.tech/projects/signalforge)

**Option 1: Full HTML Page**

1. Open `docs/SIGNALFORGE_PROJECT_PAGE.html` from your SignalForge repo
2. Copy the entire `<body>` section
3. Paste into your CMS's HTML editor
4. Adjust CSS classes to match your site's design
5. Update any hardcoded URLs
6. Preview and publish

**Option 2: Markdown/Content Sections**

Copy sections from `docs/WEBSITE_UPDATE_FINAL.md`:

1. **Hero Section** - Stats badges and key metrics
2. **Project Overview** - What this is, why it's different
3. **Model Performance** - Tables with confidence intervals
4. **Statistical Rigor** - Methodology explanations
5. **Business Impact** - ROI calculations
6. **Feature Engineering** - Learned weights section
7. **Key Learnings** - What you learned

---

## 🔧 How to Update (Step-by-Step)

### If Using WordPress/Webflow/Squarespace:

1. **Log into your CMS admin panel**
2. **Navigate to Pages > Projects**
3. **Edit the SignalForge page**
4. **Switch to HTML/Code view**
5. **Copy content from:**
   - `SIGNALFORGE_PROJECT_PAGE.html` for full page
   - `WEBSITE_UPDATE_FINAL.md` for individual sections
6. **Paste into your editor**
7. **Adjust styling as needed**
8. **Preview on mobile**
9. **Save and publish**

### If Using Static Site Generator (Hugo/Jekyll/Next.js):

1. **Navigate to your website repo locally**
2. **Find the SignalForge project file** (usually in `content/projects/` or `pages/`)
3. **Replace the frontmatter and content**
4. **Commit and push**
5. **Wait for deployment**

---

## 📊 Key Metrics to Highlight

Make sure these are prominently displayed:

### Hero Section:
```
0.850 ± 0.013 AUC (5-fold CV)
p=0.0074 (vs Random Forest)
$1.67M revenue at risk
1.21x - 1.81x expected ROI
```

### Performance Table:
| Model | AUC | 95% CI | Significance |
|-------|-----|--------|--------------|
| Logistic Regression | 0.850 | [0.827, 0.870] | p=0.0074 ✅ |
| Random Forest | 0.839 | [0.821, 0.857] | - |
| Gradient Boosting | 0.832 | [0.812, 0.852] | - |

### Calibration:
| Model | Brier Score | ECE |
|-------|-------------|-----|
| Gradient Boosting | 0.139 | 0.033 |
| Logistic Regression | 0.164 | 0.147 |

---

## 🎨 Design Tips

### Badges/Tags:
- Add "Statistical Rigor ✅" badge
- Add "Production Ready" badge
- Add confidence intervals as tooltips
- Use green checkmarks for significant results

### Visual Hierarchy:
1. Hero stats (big numbers)
2. Key findings (what matters)
3. Detailed methodology (for technical readers)
4. Business impact (for stakeholders)

### Color Coding:
- Green: Significant/winning model
- Blue: Confidence intervals
- Orange: Warnings/trade-offs
- Red: At-risk metrics

---

## ✅ Pre-Publish Checklist

Before publishing, verify:

- [ ] All confidence intervals are displayed
- [ ] p-values are shown for model comparisons
- [ ] Calibration analysis is included
- [ ] Business impact ($1.67M) is prominent
- [ ] ROI calculations (1.21x - 1.81x) are clear
- [ ] Learned feature weights are explained
- [ ] All links work (GitHub, dashboard, etc.)
- [ ] Mobile responsive
- [ ] SEO meta tags updated
- [ ] Schema.org markup added (if supported)

---

## 🔍 SEO Updates

**Update your meta tags:**

```html
<title>SignalForge - Production Churn Intelligence with Statistical Rigor | Christian Callahan</title>
<meta name="description" content="ML system achieving 0.850 ± 0.013 AUC with statistical significance, identifying $1.67M revenue at risk. Demonstrates bootstrap confidence intervals, significance testing, and calibration analysis.">
<meta name="keywords" content="churn prediction, machine learning, statistical rigor, confidence intervals, significance testing, calibration analysis, cross-validation">
```

**Add Schema.org markup:**
```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "SignalForge",
  "description": "Production churn intelligence with statistical rigor",
  "author": {"@type": "Person", "name": "Christian Callahan"},
  "codeRepository": "https://github.com/CCallahan308/signalforge"
}
```

---

## 📱 Mobile Considerations

**Test on mobile:**
- Tables should be scrollable horizontally
- Stats grid should stack vertically
- Long numbers should wrap properly
- Code blocks should be readable

---

## 🚀 After Publishing

1. **Clear your browser cache**
2. **Test on multiple devices**
3. **Check all links work**
4. **Verify social sharing** (Open Graph tags)
5. **Test page speed**
6. **Submit updated sitemap** to Google

---

## 💬 Need Help?

If you run into issues:

1. **CMS-specific problems:** Check your platform's documentation
2. **Styling issues:** Use browser dev tools to debug
3. **Content formatting:** Most CMSs have a preview feature
4. **SEO validation:** Use Google's Rich Results Test

---

## 📁 Quick File Reference

All files are in: `C:\Users\Calla\signalforge\docs\`

| File | Purpose | Format |
|------|---------|--------|
| `WEBSITE_HOMEPAGE_UPDATE.md` | Homepage content | Markdown |
| `SIGNALFORGE_PROJECT_PAGE.html` | Full project page | HTML |
| `WEBSITE_UPDATE_FINAL.md` | All sections + SEO | Markdown |
| `WEBSITE_UPDATE_SHORT.md` | Projects grid version | Markdown |

---

## ✅ You're Ready!

1. **Copy the files** from `signalforge/docs/`
2. **Paste into your CMS**
3. **Adjust styling**
4. **Preview on mobile**
5. **Publish!**

**Your website will now showcase:**
- ✅ Statistical rigor (confidence intervals, p-values)
- ✅ Business impact ($1.67M revenue at risk)
- ✅ Production quality (Docker, PostgreSQL)
- ✅ Depth (calibration, regularization, feature learning)

**This positions you for:**
- $200-280K tech roles
- $300-500K quant roles
- Senior data scientist positions

---

**Need me to create any specific format or section? Just let me know! 🎯**
