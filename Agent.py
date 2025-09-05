import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass, asdict
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic AI Matching System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.match-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.agent-info {
    background: #f8f9ff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 3px solid #4CAF50;
    margin: 1rem 0;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}
</style>
""", unsafe_allow_html=True)

# Data Models
@dataclass
class Employee:
    id: str
    name: str
    email: str
    skills: List[str]
    experience_years: int
    current_role: str
    availability: str  # Available, Busy, Partial
    career_interests: List[str]
    location: str
    performance_rating: float
    last_project_end: str
    salary_range: str = "Not Specified"
    certifications: List[str] = None
    
    def __post_init__(self):
        if self.certifications is None:
            self.certifications = []
    
@dataclass
class Project:
    id: str
    name: str
    description: str
    required_skills: List[str]
    duration_months: int
    urgency: str  # High, Medium, Low
    location: str
    team_size: int
    experience_required: int
    budget_category: str
    client_type: str
    start_date: str
    priority_score: float = 1.0
    
@dataclass
class Match:
    employee_id: str
    project_id: str
    match_score: float
    skill_match: float
    availability_match: float
    experience_match: float
    location_match: float
    reasoning: str
    created_at: str
    confidence_level: str = "Medium"

# AI Agent Classes
class SkillMatchingAgent:
    """AI Agent responsible for skill-based matching"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.name = "Skill Matching Agent"
        
    def calculate_skill_match(self, employee_skills: List[str], required_skills: List[str]) -> Tuple[float, str]:
        """Calculate skill match score between employee and project"""
        if not employee_skills or not required_skills:
            return 0.0, "❌ Insufficient skill data"
        
        # Combine skills into text for TF-IDF
        emp_text = " ".join(employee_skills).lower()
        req_text = " ".join(required_skills).lower()
        
        try:
            # Calculate TF-IDF similarity
            texts = [emp_text, req_text]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Calculate direct skill overlap
            emp_skills_set = set(skill.lower().strip() for skill in employee_skills)
            req_skills_set = set(skill.lower().strip() for skill in required_skills)
            
            overlap = len(emp_skills_set.intersection(req_skills_set))
            total_required = len(req_skills_set)
            direct_match = overlap / total_required if total_required > 0 else 0
            
            # Combine both scores with weights
            final_score = (similarity * 0.6) + (direct_match * 0.4)
            
            # Enhanced reasoning
            matched_skills = emp_skills_set.intersection(req_skills_set)
            missing_skills = req_skills_set - emp_skills_set
            
            reasoning = f"✅ Direct match: {overlap}/{total_required} skills"
            if matched_skills:
                reasoning += f" | Matched: {', '.join(list(matched_skills)[:3])}"
            if missing_skills and len(missing_skills) <= 3:
                reasoning += f" | Missing: {', '.join(list(missing_skills))}"
            reasoning += f" | Semantic similarity: {similarity:.2f}"
            
            return min(final_score, 1.0), reasoning
            
        except Exception as e:
            return 0.0, f"❌ Error in skill matching: {str(e)}"

class AvailabilityAgent:
    """AI Agent for availability and timing matching"""
    
    def __init__(self):
        self.name = "Availability Agent"
        
    def calculate_availability_match(self, employee: Employee, project: Project) -> Tuple[float, str]:
        """Calculate availability match score"""
        availability_scores = {
            "Available": 1.0,
            "Partial": 0.6,
            "Busy": 0.2
        }
        
        base_score = availability_scores.get(employee.availability, 0.0)
        
        # Consider last project end date
        try:
            last_end = datetime.strptime(employee.last_project_end, "%Y-%m-%d")
            project_start = datetime.strptime(project.start_date, "%Y-%m-%d")
            days_gap = (project_start - last_end).days
            
            if days_gap >= 30:  # 1 month gap
                gap_bonus = 0.2
                gap_desc = "🟢 Excellent gap"
            elif days_gap >= 7:  # 1 week gap
                gap_bonus = 0.1
                gap_desc = "🟡 Good gap"
            elif days_gap >= 0:
                gap_bonus = 0.0
                gap_desc = "🟠 Tight timing"
            else:
                gap_bonus = -0.1
                gap_desc = "🔴 Overlap conflict"
                
            final_score = max(0.0, min(base_score + gap_bonus, 1.0))
            reasoning = f"📅 Status: {employee.availability} | {gap_desc} ({days_gap} days)"
            
        except Exception:
            final_score = base_score
            reasoning = f"📅 Status: {employee.availability} | Date parsing error"
            
        return final_score, reasoning

class ExperienceAgent:
    """AI Agent for experience level matching"""
    
    def __init__(self):
        self.name = "Experience Matching Agent"
        
    def calculate_experience_match(self, employee: Employee, project: Project) -> Tuple[float, str]:
        """Calculate experience match score"""
        emp_exp = employee.experience_years
        req_exp = project.experience_required
        
        if emp_exp >= req_exp:
            # Employee meets or exceeds requirements
            if emp_exp <= req_exp + 2:
                score = 1.0  # Perfect match
                level = "🎯 Perfect match"
            elif emp_exp <= req_exp + 5:
                score = 0.9  # Slightly overqualified
                level = "🟢 Well qualified"
            else:
                # Significantly overqualified - might be bored or expensive
                excess = emp_exp - req_exp
                score = max(0.6, 1.0 - (excess * 0.03))
                level = "🟡 Overqualified"
        else:
            # Underqualified
            gap = req_exp - emp_exp
            if gap <= 1:
                score = 0.8
                level = "🟠 Close match"
            elif gap <= 2:
                score = 0.6
                level = "🔴 Some gap"
            else:
                score = max(0.2, 1.0 - (gap * 0.15))
                level = "❌ Significant gap"
            
        reasoning = f"👨‍💼 Employee: {emp_exp} years | Required: {req_exp} years | {level}"
        return score, reasoning

class LocationAgent:
    """AI Agent for location and remote work matching"""
    
    def __init__(self):
        self.name = "Location Matching Agent"
        
    def calculate_location_match(self, employee: Employee, project: Project) -> Tuple[float, str]:
        """Calculate location compatibility score"""
        emp_loc = employee.location.lower().strip()
        proj_loc = project.location.lower().strip()
        
        if emp_loc == proj_loc:
            return 1.0, f"🎯 Same location: {employee.location}"
        elif "remote" in proj_loc or "remote" in emp_loc:
            return 0.95, "🏠 Remote work compatible"
        elif any(city in emp_loc and city in proj_loc for city in 
                ["bangalore", "mumbai", "delhi", "hyderabad", "chennai", "pune"]):
            return 0.85, f"🏙️ Same metro area"
        elif any(region in emp_loc and region in proj_loc for region in 
                ["south", "north", "west", "east"]):
            return 0.6, "🗺️ Same region - manageable travel"
        else:
            return 0.3, f"✈️ Different locations - significant travel required"

class PerformanceAgent:
    """AI Agent for performance and quality assessment"""
    
    def __init__(self):
        self.name = "Performance Assessment Agent"
        
    def calculate_performance_bonus(self, employee: Employee) -> Tuple[float, str]:
        """Calculate performance-based bonus/penalty"""
        rating = employee.performance_rating
        
        if rating >= 4.5:
            return 0.15, "⭐ Outstanding performer"
        elif rating >= 4.0:
            return 0.10, "🌟 High performer"
        elif rating >= 3.5:
            return 0.05, "👍 Good performer"
        elif rating >= 3.0:
            return 0.0, "📊 Average performer"
        else:
            return -0.05, "⚠️ Below average performance"

class MasterMatchingAgent:
    """Master AI Agent that coordinates all other agents"""
    
    def __init__(self):
        self.skill_agent = SkillMatchingAgent()
        self.availability_agent = AvailabilityAgent()
        self.experience_agent = ExperienceAgent()
        self.location_agent = LocationAgent()
        self.performance_agent = PerformanceAgent()
        self.agents = [
            self.skill_agent, self.availability_agent, 
            self.experience_agent, self.location_agent, self.performance_agent
        ]
        
    def find_matches(self, employees: List[Employee], projects: List[Project], 
                    min_score: float = 0.6, max_per_project: int = 5) -> List[Match]:
        """Find best matches between employees and projects"""
        matches = []
        
        for project in projects:
            project_matches = []
            
            for employee in employees:
                match_result = self._calculate_match(employee, project)
                if match_result.match_score >= min_score:
                    project_matches.append(match_result)
            
            # Sort by match score and take top matches
            project_matches.sort(key=lambda x: x.match_score, reverse=True)
            matches.extend(project_matches[:max_per_project])
            
        return sorted(matches, key=lambda x: x.match_score, reverse=True)
    
    def _calculate_match(self, employee: Employee, project: Project) -> Match:
        """Calculate comprehensive match score using all agents"""
        # Get scores from each specialized agent
        skill_score, skill_reason = self.skill_agent.calculate_skill_match(
            employee.skills, project.required_skills)
        avail_score, avail_reason = self.availability_agent.calculate_availability_match(
            employee, project)
        exp_score, exp_reason = self.experience_agent.calculate_experience_match(
            employee, project)
        loc_score, loc_reason = self.location_agent.calculate_location_match(
            employee, project)
        perf_bonus, perf_reason = self.performance_agent.calculate_performance_bonus(employee)
        
        # Dynamic weights based on project urgency
        if project.urgency == "High":
            weights = {"skill": 0.45, "availability": 0.35, "experience": 0.15, "location": 0.05}
        elif project.urgency == "Medium":
            weights = {"skill": 0.4, "availability": 0.25, "experience": 0.2, "location": 0.15}
        else:  # Low urgency
            weights = {"skill": 0.35, "availability": 0.2, "experience": 0.25, "location": 0.2}
        
        overall_score = (
            skill_score * weights["skill"] +
            avail_score * weights["availability"] +
            exp_score * weights["experience"] +
            loc_score * weights["location"]
        )
        
        # Apply performance bonus
        overall_score += perf_bonus
        overall_score = max(0.0, min(overall_score, 1.0))
        
        # Determine confidence level
        if overall_score >= 0.85:
            confidence = "High"
        elif overall_score >= 0.7:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        reasoning = " | ".join([skill_reason, avail_reason, exp_reason, loc_reason, perf_reason])
        
        return Match(
            employee_id=employee.id,
            project_id=project.id,
            match_score=overall_score,
            skill_match=skill_score,
            availability_match=avail_score,
            experience_match=exp_score,
            location_match=loc_score,
            reasoning=reasoning,
            created_at=datetime.now().isoformat(),
            confidence_level=confidence
        )

# Enhanced Data Management
class DataManager:
    """Manages sample data and user inputs with enhanced datasets"""
    
    @staticmethod
    def get_sample_employees() -> List[Employee]:
        """Generate comprehensive sample employee data"""
        return [
            Employee("emp_001", "Alice Johnson", "alice@company.com", 
                    ["Python", "Machine Learning", "TensorFlow", "AWS", "Docker", "Kubernetes"], 
                    5, "Senior ML Engineer", "Available", ["AI/ML", "Cloud Architecture"], 
                    "Bangalore", 4.2, "2024-07-15", "₹15-20L", ["AWS Certified", "TensorFlow Developer"]),
            Employee("emp_002", "Bob Smith", "bob@company.com", 
                    ["Java", "Spring Boot", "Microservices", "Kubernetes", "Docker", "Redis"], 
                    7, "Tech Lead", "Partial", ["Team Leadership", "Architecture"], 
                    "Mumbai", 4.5, "2024-06-30", "₹20-25L", ["Certified Scrum Master", "AWS Solutions Architect"]),
            Employee("emp_003", "Carol Davis", "carol@company.com", 
                    ["React", "Node.js", "MongoDB", "GraphQL", "TypeScript", "AWS"], 
                    4, "Full Stack Developer", "Available", ["Frontend", "Modern Web"], 
                    "Remote", 4.0, "2024-07-20", "₹12-15L", ["React Developer", "MongoDB Certified"]),
            Employee("emp_004", "David Wilson", "david@company.com", 
                    ["DevOps", "Jenkins", "Terraform", "AWS", "Ansible", "Monitoring"], 
                    6, "DevOps Engineer", "Busy", ["Cloud Infrastructure", "Automation"], 
                    "Hyderabad", 4.3, "2024-05-15", "₹18-22L", ["AWS DevOps", "Terraform Associate"]),
            Employee("emp_005", "Emma Brown", "emma@company.com", 
                    ["Data Science", "Python", "TensorFlow", "SQL", "Tableau", "Spark"], 
                    3, "Data Scientist", "Available", ["AI Research", "Analytics"], 
                    "Pune", 4.1, "2024-07-25", "₹10-14L", ["Tableau Desktop", "Google Data Analytics"]),
            Employee("emp_006", "Frank Miller", "frank@company.com", 
                    ["Flutter", "React Native", "iOS", "Android", "Firebase"], 
                    5, "Mobile Developer", "Available", ["Mobile Apps", "Cross-platform"], 
                    "Bangalore", 3.9, "2024-07-10", "₹14-18L", ["Flutter Certified", "iOS Developer"]),
            Employee("emp_007", "Grace Lee", "grace@company.com", 
                    ["UI/UX Design", "Figma", "Adobe XD", "Prototyping", "User Research"], 
                    4, "Senior Designer", "Partial", ["Design Systems", "User Experience"], 
                    "Remote", 4.4, "2024-06-20", "₹12-16L", ["Adobe Certified", "Google UX Design"]),
        ]
    
    @staticmethod
    def get_sample_projects() -> List[Project]:
        """Generate comprehensive sample project data"""
        return [
            Project("proj_001", "AI-Powered E-commerce Recommendations", 
                   "Build advanced ML recommendation system with real-time personalization", 
                   ["Python", "Machine Learning", "TensorFlow", "AWS", "Redis"], 6, "High", 
                   "Bangalore", 4, 4, "High", "E-commerce", "2024-08-01", 1.2),
            Project("proj_002", "Banking Microservices Migration", 
                   "Migrate legacy banking system to cloud-native microservices", 
                   ["Java", "Spring Boot", "Microservices", "Docker", "Kubernetes"], 8, "Medium", 
                   "Mumbai", 6, 5, "High", "Banking", "2024-08-15", 1.0),
            Project("proj_003", "Healthcare Customer Portal", 
                   "Modern React-based patient portal with real-time features", 
                   ["React", "Node.js", "MongoDB", "TypeScript", "WebSocket"], 4, "Medium", 
                   "Remote", 3, 3, "Medium", "Healthcare", "2024-09-01", 0.8),
            Project("proj_004", "Startup Cloud Infrastructure", 
                   "Setup scalable cloud infrastructure with CI/CD pipelines", 
                   ["DevOps", "AWS", "Terraform", "Jenkins", "Kubernetes"], 5, "High", 
                   "Hyderabad", 2, 4, "Medium", "Startup", "2024-08-20", 1.1),
            Project("proj_005", "Business Intelligence Dashboard", 
                   "Create comprehensive analytics dashboard for executive decision making", 
                   ["Data Science", "Python", "Tableau", "SQL", "Spark"], 3, "Low", 
                   "Pune", 2, 3, "Medium", "Enterprise", "2024-09-15", 0.7),
            Project("proj_006", "Mobile Banking App", 
                   "Cross-platform mobile banking application with biometric security", 
                   ["Flutter", "React Native", "Firebase", "Security"], 7, "High", 
                   "Mumbai", 5, 4, "High", "Banking", "2024-08-10", 1.3),
        ]

# Enhanced UI Components
def render_header():
    """Render attractive header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic AI Matching System</h1>
        <p>Intelligent Employee-Project Matching Platform powered by Specialized AI Agents</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Enhanced sidebar with agent information"""
    st.sidebar.title("🚀 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Dashboard", "🎯 AI Matching", "👥 Employees", "📋 Projects", "📊 Analytics", "🤖 AI Agents Info"]
    )
    
    st.sidebar.markdown("---")
    
    # Agent status indicator
    st.sidebar.markdown("### 🤖 AI Agents Status")
    agents = ["Skill Matching", "Availability", "Experience", "Location", "Performance"]
    for agent in agents:
        st.sidebar.markdown(f"🟢 {agent} Agent: *Active*")
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **About this System**\n\n"
        "This platform uses 5 specialized AI agents working together to find optimal matches between employees and projects.\n\n"
        "Each agent analyzes different aspects and the Master Agent combines their insights for the best recommendations."
    )
    
    return page

def render_dashboard():
    """Enhanced dashboard with better visualizations"""
    render_header()
    
    # Load data
    employees = st.session_state.get('employees', DataManager.get_sample_employees())
    projects = st.session_state.get('projects', DataManager.get_sample_projects())
    
    # Enhanced metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Total Employees", len(employees))
    with col2:
        st.metric("📋 Active Projects", len(projects))
    with col3:
        available_count = len([e for e in employees if e.availability == "Available"])
        st.metric("✅ Available", available_count)
    with col4:
        urgent_count = len([p for p in projects if p.urgency == "High"])
        st.metric("🚨 Urgent Projects", urgent_count)
    with col5:
        high_performers = len([e for e in employees if e.performance_rating >= 4.0])
        st.metric("⭐ Top Performers", high_performers)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Run AI Matching", type="primary", use_container_width=True):
            st.switch_page("🎯 AI Matching")
    with col2:
        if st.button("👥 Add New Employee", use_container_width=True):
            st.switch_page("👥 Employees")
    with col3:
        if st.button("📋 Create New Project", use_container_width=True):
            st.switch_page("📋 Projects")
    
    st.markdown("---")
    
    # Enhanced visualizations
    st.markdown("### 📊 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Availability distribution
        avail_data = {}
        for emp in employees:
            avail_data[emp.availability] = avail_data.get(emp.availability, 0) + 1
        
        colors = {'Available': '#4CAF50', 'Partial': '#FF9800', 'Busy': '#F44336'}
        fig = px.pie(values=list(avail_data.values()), names=list(avail_data.keys()), 
                    title="Employee Availability Distribution",
                    color_discrete_map=colors)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Project urgency with custom colors
        urgency_data = {}
        for proj in projects:
            urgency_data[proj.urgency] = urgency_data.get(proj.urgency, 0) + 1
        
        urgency_colors = {'High': '#F44336', 'Medium': '#FF9800', 'Low': '#4CAF50'}
        fig = px.bar(x=list(urgency_data.keys()), y=list(urgency_data.values()),
                    title="Project Urgency Distribution",
                    color=list(urgency_data.keys()),
                    color_discrete_map=urgency_colors)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance vs Experience scatter
    st.markdown("### 👥 Employee Performance Analysis")
    
    emp_df = pd.DataFrame([
        {
            'name': emp.name,
            'experience': emp.experience_years,
            'performance': emp.performance_rating,
            'location': emp.location,
            'availability': emp.availability
        } for emp in employees
    ])
    
    fig = px.scatter(emp_df, x='experience', y='performance', 
                     size='performance', color='availability',
                     hover_data=['name', 'location'],
                     title="Experience vs Performance Rating",
                     labels={'experience': 'Years of Experience', 
                            'performance': 'Performance Rating'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_matching_page():
    """Enhanced AI matching interface"""
    st.title("🎯 AI-Powered Matching System")
    st.markdown("### Let our AI agents find the perfect matches!")
    
    # Load data
    employees = st.session_state.get('employees', DataManager.get_sample_employees())
    projects = st.session_state.get('projects', DataManager.get_sample_projects())
    
    # Enhanced configuration
    st.markdown("### 🔧 Matching Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("🎯 Minimum Match Score", 0.0, 1.0, 0.6, 0.05,
                             help="Only show matches above this score")
        max_results = st.selectbox("📊 Max Results per Project", [3, 5, 8, 10], index=1)
    
    with col2:
        project_filter = st.selectbox("📋 Filter by Project", 
                                    ["All Projects"] + [p.name for p in projects])
        urgency_filter = st.selectbox("🚨 Filter by Urgency", 
                                    ["All", "High", "Medium", "Low"])
    
    with col3:
        availability_filter = st.selectbox("✅ Employee Availability", 
                                         ["All", "Available", "Partial", "Busy"])
        show_reasoning = st.checkbox("🧠 Show AI Reasoning", value=True)
    
    st.markdown("---")
    
    # AI Agent Information
    with st.expander("🤖 AI Agents Working on This Task"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Active AI Agents:**
            - 🎯 **Skill Matching Agent**: Analyzes technical skills and requirements
            - 📅 **Availability Agent**: Checks timing and schedule compatibility  
            - 👨‍💼 **Experience Agent**: Evaluates experience levels and career fit
            """)
        with col2:
            st.markdown("""
            **Supporting Agents:**
            - 🗺️ **Location Agent**: Handles geographic and remote work preferences
            - ⭐ **Performance Agent**: Considers past performance ratings
            - 🧠 **Master Agent**: Coordinates all agents for final recommendations
            """)
    
    if st.button("🚀 Run AI Matching Analysis", type="primary", use_container_width=True):
        # Progress bar for better UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('🤖 Initializing AI agents...')
        progress_bar.progress(20)
        
        # Initialize master agent
        master_agent = MasterMatchingAgent()
        
        status_text.text('🔍 Filtering data based on criteria...')
        progress_bar.progress(40)
        
        # Apply filters
        filtered_projects = projects
        filtered_employees = employees
        
        if project_filter != "All Projects":
            filtered_projects = [p for p in projects if p.name == project_filter]
        if urgency_filter != "All":
            filtered_projects = [p for p in filtered_projects if p.urgency == urgency_filter]
        if availability_filter != "All":
            filtered_employees = [e for e in employees if e.availability == availability_filter]
        
        status_text.text('🧠 AI agents analyzing matches...')
        progress_bar.progress(70)
        
        # Find matches
        matches = master_agent.find_matches(filtered_employees, filtered_projects, min_score)
        
        status_text.text('✅ Analysis complete!')
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        if matches:
            st.success(f"🎉 Found {len(matches)} high-quality matches!")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = sum(m.match_score for m in matches) / len(matches)
                st.metric("Average Score", f"{avg_score:.2f}")
            with col2:
                high_confidence = len([m for m in matches if m.confidence_level == "High"])
                st.metric("High Confidence", high_confidence)
            with col3:
                perfect_matches = len([m for m in matches if m.match_score >= 0.9])
                st.metric("Perfect Matches", perfect_matches)
            with col4:
                unique_employees = len(set(m.employee_id for m in matches))
                st.metric("Employees Matched", unique_employees)
            
            st.markdown("### 🎯 Match Results")
            
            # Display matches in groups by project
            current_project = None
            for i, match in enumerate(matches[:max_results * len(filtered_projects)]):
                employee = next(e for e in employees if e.id == match.employee_id)
                project = next(p for p in projects if p.id == match.project_id)
                
                # Project header
                if current_project != project.id:
                    current_project = project.id
                    st.markdown(f"#### 📋 {project.name}")
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"*{project.description}*")
                    with col2:
                        st.markdown(f"**Urgency:** {project.urgency}")
                    with col3:
                        st.markdown(f"**Duration:** {project.duration_months}m")
                
                # Match card
                with st.container():
                    st.markdown(f"""
                    <div class="match-card">
                        <h4>🏆 Match #{i+1}: {employee.name} 
                        <span style="float:right; color: {'#4CAF50' if match.confidence_level == 'High' else '#FF9800' if match.confidence_level == 'Medium' else '#F44336'}">
                            {match.match_score:.2f} ({match.confidence_level} Confidence)
                        </span></h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**👤 Employee Profile:**")
                        st.write(f"**Name:** {employee.name}")
                        st.write(f"**Role:** {employee.current_role}")
                        st.write(f"**Experience:** {employee.experience_years} years")
                        st.write(f"**Skills:** {', '.join(employee.skills[:4])}{'...' if len(employee.skills) > 4 else ''}")
                        st.write(f"**Availability:** {employee.availability}")
                        st.write(f"**Location:** {employee.location}")
                        st.write(f"**Performance Rating:** ⭐ {employee.performance_rating}/5.0")
                        if employee.certifications:
                            st.write(f"**Certifications:** {', '.join(employee.certifications[:2])}{'...' if len(employee.certifications) > 2 else ''}")
                    
                    with col2:
                        st.markdown("**📋 Project Requirements:**")
                        st.write(f"**Duration:** {project.duration_months} months")
                        st.write(f"**Required Skills:** {', '.join(project.required_skills)}")
                        st.write(f"**Experience Needed:** {project.experience_required} years")
                        st.write(f"**Team Size:** {project.team_size} members")
                        st.write(f"**Location:** {project.location}")
                        st.write(f"**Client Type:** {project.client_type}")
                        st.write(f"**Budget:** {project.budget_category}")
                    
                    # Detailed scoring
                    st.markdown("**🤖 AI Agent Analysis:**")
                    col3, col4, col5, col6 = st.columns(4)
                    
                    def get_score_color(score):
                        if score >= 0.8: return "#4CAF50"
                        elif score >= 0.6: return "#FF9800"
                        else: return "#F44336"
                    
                    with col3:
                        color = get_score_color(match.skill_match)
                        st.markdown(f"<div style='text-align:center'><h4 style='color:{color}'>{match.skill_match:.2f}</h4><p>Skill Match</p></div>", unsafe_allow_html=True)
                    with col4:
                        color = get_score_color(match.availability_match)
                        st.markdown(f"<div style='text-align:center'><h4 style='color:{color}'>{match.availability_match:.2f}</h4><p>Availability</p></div>", unsafe_allow_html=True)
                    with col5:
                        color = get_score_color(match.experience_match)
                        st.markdown(f"<div style='text-align:center'><h4 style='color:{color}'>{match.experience_match:.2f}</h4><p>Experience</p></div>", unsafe_allow_html=True)
                    with col6:
                        color = get_score_color(match.location_match)
                        st.markdown(f"<div style='text-align:center'><h4 style='color:{color}'>{match.location_match:.2f}</h4><p>Location</p></div>", unsafe_allow_html=True)
                    
                    # AI Reasoning
                    if show_reasoning:
                        st.markdown("**🧠 AI Agent Reasoning:**")
                        reasoning_parts = match.reasoning.split(" | ")
                        for part in reasoning_parts:
                            st.markdown(f"• {part}")
                    
                    # Action buttons
                    col7, col8, col9 = st.columns([1, 1, 2])
                    with col7:
                        if st.button(f"✅ Assign", key=f"assign_{i}", type="primary"):
                            st.success("✅ Assignment request sent to project manager!")
                            st.balloons()
                    with col8:
                        if st.button(f"📧 Contact", key=f"contact_{i}"):
                            st.info(f"📧 Opening email to {employee.email}")
                    with col9:
                        if st.button(f"📊 View Full Profile", key=f"profile_{i}"):
                            st.info("Redirecting to detailed employee profile...")
                    
                    st.markdown("---")
        else:
            st.warning("🔍 No matches found with current criteria. Try adjusting the filters or lowering the minimum score.")
            
            # Suggestions
            st.markdown("### 💡 Suggestions:")
            st.markdown("""
            - Lower the minimum match score
            - Remove some filters to broaden the search
            - Check if there are available employees
            - Consider projects with different urgency levels
            """)

def render_employee_management():
    """Enhanced employee management with better forms"""
    st.title("👥 Employee Management Hub")
    
    # Initialize employees in session state
    if 'employees' not in st.session_state:
        st.session_state.employees = DataManager.get_sample_employees()
    
    tab1, tab2, tab3 = st.tabs(["📋 View All Employees", "➕ Add New Employee", "📊 Employee Analytics"])
    
    with tab1:
        st.markdown("### Current Employee Database")
        
        # Search and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search by name or skills", "")
        with col2:
            location_filter = st.selectbox("📍 Filter by location", 
                ["All Locations"] + list(set(emp.location for emp in st.session_state.employees)))
        with col3:
            availability_filter = st.selectbox("✅ Filter by availability", 
                ["All", "Available", "Partial", "Busy"])
        
        # Filter employees
        filtered_employees = st.session_state.employees
        
        if search_term:
            filtered_employees = [emp for emp in filtered_employees 
                                if search_term.lower() in emp.name.lower() or 
                                   any(search_term.lower() in skill.lower() for skill in emp.skills)]
        
        if location_filter != "All Locations":
            filtered_employees = [emp for emp in filtered_employees if emp.location == location_filter]
            
        if availability_filter != "All":
            filtered_employees = [emp for emp in filtered_employees if emp.availability == availability_filter]
        
        # Display enhanced employee cards
        for emp in filtered_employees:
            with st.expander(f"👤 {emp.name} - {emp.current_role}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Basic Information:**")
                    st.write(f"**Email:** {emp.email}")
                    st.write(f"**Experience:** {emp.experience_years} years")
                    st.write(f"**Location:** {emp.location}")
                    st.write(f"**Availability:** {emp.availability}")
                    st.write(f"**Performance:** ⭐ {emp.performance_rating}/5.0")
                    st.write(f"**Salary Range:** {emp.salary_range}")
                
                with col2:
                    st.markdown("**🛠️ Skills & Expertise:**")
                    # Display skills as badges
                    skills_html = ""
                    for skill in emp.skills:
                        skills_html += f'<span style="background:#e1f5fe; padding:4px 8px; margin:2px; border-radius:12px; font-size:12px;">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    st.markdown("**🎯 Career Interests:**")
                    for interest in emp.career_interests:
                        st.write(f"• {interest}")
                    
                    if emp.certifications:
                        st.markdown("**📜 Certifications:**")
                        for cert in emp.certifications:
                            st.write(f"🏆 {cert}")
    
    with tab2:
        st.markdown("### Add New Team Member")
        
        with st.form("add_employee", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("👤 Full Name *", placeholder="e.g., John Smith")
                email = st.text_input("📧 Email Address *", placeholder="john.smith@company.com")
                current_role = st.text_input("💼 Current Role *", placeholder="e.g., Senior Developer")
                experience = st.slider("📈 Years of Experience", 0, 30, 3)
                performance = st.slider("⭐ Performance Rating", 1.0, 5.0, 3.5, 0.1)
            
            with col2:
                location = st.selectbox("📍 Location", 
                    ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Remote", "Other"])
                availability = st.selectbox("✅ Current Availability", 
                    ["Available", "Partial", "Busy"])
                salary_range = st.selectbox("💰 Salary Range", 
                    ["₹5-8L", "₹8-12L", "₹12-15L", "₹15-20L", "₹20-25L", "₹25L+", "Not Specified"])
                last_project = st.date_input("📅 Last Project End Date", 
                    value=datetime.now() - timedelta(days=30))
            
            skills = st.text_area("🛠️ Technical Skills *", 
                placeholder="Python, Machine Learning, AWS, Docker, Kubernetes",
                help="Enter skills separated by commas")
            
            interests = st.text_area("🎯 Career Interests", 
                placeholder="AI/ML, Cloud Architecture, Team Leadership",
                help="Enter career interests separated by commas")
            
            certifications = st.text_area("📜 Certifications", 
                placeholder="AWS Certified Solutions Architect, Scrum Master",
                help="Enter certifications separated by commas")
            
            col3, col4 = st.columns(2)
            with col3:
                submitted = st.form_submit_button("✅ Add Employee", type="primary", use_container_width=True)
            with col4:
                st.form_submit_button("🔄 Clear Form", use_container_width=True)
            
            if submitted:
                if name and email and current_role and skills:
                    new_employee = Employee(
                        id=f"emp_{str(uuid.uuid4())[:8]}",
                        name=name,
                        email=email,
                        skills=[s.strip() for s in skills.split(",") if s.strip()],
                        experience_years=experience,
                        current_role=current_role,
                        availability=availability,
                        career_interests=[i.strip() for i in interests.split(",") if i.strip()],
                        location=location,
                        performance_rating=performance,
                        last_project_end=last_project.strftime("%Y-%m-%d"),
                        salary_range=salary_range,
                        certifications=[c.strip() for c in certifications.split(",") if c.strip()]
                    )
                    st.session_state.employees.append(new_employee)
                    st.success(f"🎉 {name} has been successfully added to the team!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields marked with *")
    
    with tab3:
        st.markdown("### Employee Analytics Dashboard")
        
        employees = st.session_state.employees
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_experience = sum(emp.experience_years for emp in employees) / len(employees)
            st.metric("📈 Avg Experience", f"{avg_experience:.1f} years")
        with col2:
            avg_performance = sum(emp.performance_rating for emp in employees) / len(employees)
            st.metric("⭐ Avg Performance", f"{avg_performance:.1f}/5.0")
        with col3:
            available_count = len([e for e in employees if e.availability == "Available"])
            st.metric("✅ Available Now", f"{available_count}/{len(employees)}")
        with col4:
            certified_count = len([e for e in employees if e.certifications])
            st.metric("📜 With Certifications", f"{certified_count}/{len(employees)}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Experience distribution
            exp_data = [emp.experience_years for emp in employees]
            fig = px.histogram(x=exp_data, nbins=10, title="Experience Distribution",
                             labels={'x': 'Years of Experience', 'y': 'Number of Employees'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance vs Experience
            perf_exp_data = [(emp.experience_years, emp.performance_rating, emp.name) for emp in employees]
            df = pd.DataFrame(perf_exp_data, columns=['Experience', 'Performance', 'Name'])
            fig = px.scatter(df, x='Experience', y='Performance', hover_data=['Name'],
                           title="Performance vs Experience")
            st.plotly_chart(fig, use_container_width=True)
        
        # Top skills analysis
        st.markdown("### 🛠️ Skills Analysis")
        skill_counts = {}
        for emp in employees:
            for skill in emp.skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        if skill_counts:
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
            fig = px.bar(skills_df, x='Count', y='Skill', orientation='h',
                        title="Most Common Skills in Team")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def render_project_management():
    """Enhanced project management interface"""
    st.title("📋 Project Management Hub")
    
    # Initialize projects in session state
    if 'projects' not in st.session_state:
        st.session_state.projects = DataManager.get_sample_projects()
    
    tab1, tab2, tab3 = st.tabs(["📊 Active Projects", "🆕 Create Project", "📈 Project Analytics"])
    
    with tab1:
        st.markdown("### Active Project Portfolio")
        
        projects = st.session_state.projects
        
        # Filter and search options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search projects", "")
        with col2:
            urgency_filter = st.selectbox("🚨 Filter by urgency", ["All", "High", "Medium", "Low"])
        with col3:
            client_filter = st.selectbox("👥 Filter by client type", 
                ["All"] + list(set(proj.client_type for proj in projects)))
        
        # Apply filters
        filtered_projects = projects
        if search_term:
            filtered_projects = [proj for proj in filtered_projects 
                               if search_term.lower() in proj.name.lower() or 
                                  search_term.lower() in proj.description.lower()]
        if urgency_filter != "All":
            filtered_projects = [proj for proj in filtered_projects if proj.urgency == urgency_filter]
        if client_filter != "All":
            filtered_projects = [proj for proj in filtered_projects if proj.client_type == client_filter]
        
        # Display project cards
        for proj in filtered_projects:
            urgency_colors = {"High": "#ffebee", "Medium": "#fff3e0", "Low": "#e8f5e8"}
            urgency_text_colors = {"High": "#c62828", "Medium": "#ef6c00", "Low": "#2e7d32"}
            
            with st.container():
                st.markdown(f"""
                <div style="background: {urgency_colors[proj.urgency]}; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid {urgency_text_colors[proj.urgency]};">
                    <h4 style="color: {urgency_text_colors[proj.urgency]}; margin: 0;">{proj.name}</h4>
                    <p style="margin: 0.5rem 0; color: #666;">{proj.description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Project Details:**")
                    st.write(f"**Duration:** {proj.duration_months} months")
                    st.write(f"**Team Size:** {proj.team_size} members")
                    st.write(f"**Start Date:** {proj.start_date}")
                    st.write(f"**Location:** {proj.location}")
                    st.write(f"**Experience Required:** {proj.experience_required}+ years")
                
                with col2:
                    st.markdown("**💼 Requirements:**")
                    st.write(f"**Budget Category:** {proj.budget_category}")
                    st.write(f"**Client Type:** {proj.client_type}")
                    st.write(f"**Urgency Level:** {proj.urgency}")
                    
                    # Required skills as badges
                    st.markdown("**Required Skills:**")
                    skills_html = ""
                    for skill in proj.required_skills:
                        skills_html += f'<span style="background:#e3f2fd; color:#1976d2; padding:4px 8px; margin:2px; border-radius:12px; font-size:12px;">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                
                # Action buttons
                col3, col4, col5 = st.columns(3)
                with col3:
                    if st.button(f"🎯 Find Matches", key=f"match_{proj.id}", type="primary"):
                        st.session_state['selected_project'] = proj.name
                        st.info("Redirecting to AI Matching...")
                with col4:
                    if st.button(f"✏️ Edit Project", key=f"edit_{proj.id}"):
                        st.info("Edit functionality coming soon!")
                with col5:
                    if st.button(f"📈 View Analytics", key=f"analytics_{proj.id}"):
                        st.info("Project-specific analytics coming soon!")
                
                st.markdown("---")
    
    with tab2:
        st.markdown("### Create New Project")
        
        with st.form("add_project", clear_on_submit=True):
            st.markdown("#### 📋 Basic Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("📝 Project Name *", placeholder="e.g., E-commerce AI Platform")
                client_type = st.selectbox("👥 Client Type *", 
                    ["Enterprise", "Startup", "Government", "Healthcare", "Banking", "E-commerce", "Other"])
                urgency = st.selectbox("🚨 Urgency Level *", ["High", "Medium", "Low"])
                budget_category = st.selectbox("💰 Budget Category *", ["High", "Medium", "Low"])
            
            with col2:
                duration = st.slider("⏱️ Duration (months)", 1, 24, 6)
                team_size = st.slider("👥 Team Size", 1, 20, 4)
                experience_req = st.slider("📈 Min Experience Required (years)", 0, 15, 3)
                start_date = st.date_input("📅 Expected Start Date", 
                    value=datetime.now() + timedelta(days=30))
            
            description = st.text_area("📋 Project Description *", 
                placeholder="Detailed description of project goals, scope, and deliverables...",
                height=100)
            
            st.markdown("#### 🛠️ Technical Requirements")
            col3, col4 = st.columns(2)
            
            with col3:
                required_skills = st.text_area("💻 Required Technical Skills *", 
                    placeholder="Python, React, AWS, Docker, Machine Learning",
                    help="Enter skills separated by commas")
                location = st.selectbox("📍 Project Location *", 
                    ["Remote", "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Hybrid"])
            
            with col4:
                nice_to_have = st.text_area("✨ Nice-to-have Skills", 
                    placeholder="GraphQL, Kubernetes, Microservices",
                    help="Additional skills that would be beneficial")
                priority_score = st.slider("⭐ Priority Score", 0.1, 2.0, 1.0, 0.1,
                    help="Higher values indicate higher priority projects")
            
            # Form submission
            col5, col6 = st.columns(2)
            with col5:
                submitted = st.form_submit_button("🚀 Create Project", type="primary", use_container_width=True)
            with col6:
                st.form_submit_button("🔄 Clear Form", use_container_width=True)
            
            if submitted:
                if name and description and required_skills and client_type:
                    new_project = Project(
                        id=f"proj_{str(uuid.uuid4())[:8]}",
                        name=name,
                        description=description,
                        required_skills=[s.strip() for s in required_skills.split(",") if s.strip()],
                        duration_months=duration,
                        urgency=urgency,
                        location=location,
                        team_size=team_size,
                        experience_required=experience_req,
                        budget_category=budget_category,
                        client_type=client_type,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        priority_score=priority_score
                    )
                    st.session_state.projects.append(new_project)
                    st.success(f"🎉 Project '{name}' created successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields marked with *")
    
    with tab3:
        render_project_analytics()

def render_project_analytics():
    """Project analytics dashboard"""
    projects = st.session_state.get('projects', DataManager.get_sample_projects())
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_duration = sum(proj.duration_months for proj in projects) / len(projects)
        st.metric("📅 Avg Duration", f"{avg_duration:.1f} months")
    with col2:
        urgent_projects = len([p for p in projects if p.urgency == "High"])
        st.metric("🚨 Urgent Projects", urgent_projects)
    with col3:
        total_team_size = sum(proj.team_size for proj in projects)
        st.metric("👥 Total Team Needs", total_team_size)
    with col4:
        remote_projects = len([p for p in projects if "remote" in p.location.lower()])
        st.metric("🏠 Remote Projects", remote_projects)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Urgency distribution
        urgency_data = {}
        for proj in projects:
            urgency_data[proj.urgency] = urgency_data.get(proj.urgency, 0) + 1
        
        fig = px.pie(values=list(urgency_data.values()), names=list(urgency_data.keys()),
                    title="Project Urgency Distribution",
                    color_discrete_map={'High': '#F44336', 'Medium': '#FF9800', 'Low': '#4CAF50'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Client type distribution
        client_data = {}
        for proj in projects:
            client_data[proj.client_type] = client_data.get(proj.client_type, 0) + 1
        
        fig = px.bar(x=list(client_data.keys()), y=list(client_data.values()),
                    title="Projects by Client Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Skills demand analysis
    st.markdown("### 🛠️ Skills Demand Analysis")
    skill_demand = {}
    for proj in projects:
        for skill in proj.required_skills:
            skill_demand[skill] = skill_demand.get(skill, 0) + 1
    
    if skill_demand:
        top_skills = sorted(skill_demand.items(), key=lambda x: x[1], reverse=True)[:15]
        skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Demand'])
        fig = px.bar(skills_df, x='Demand', y='Skill', orientation='h',
                    title="Most In-Demand Skills Across Projects")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def render_analytics():
    """Enhanced analytics dashboard"""
    st.title("📊 Advanced Analytics Dashboard")
    
    employees = st.session_state.get('employees', DataManager.get_sample_employees())
    projects = st.session_state.get('projects', DataManager.get_sample_projects())
    
    # Overall system metrics
    st.markdown("### 🎯 System Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Total Employees", len(employees))
    with col2:
        st.metric("📋 Active Projects", len(projects))
    with col3:
        total_skills = len(set(skill for emp in employees for skill in emp.skills))
        st.metric("🛠️ Unique Skills", total_skills)
    with col4:
        avg_team_size = sum(proj.team_size for proj in projects) / len(projects) if projects else 0
        st.metric("👥 Avg Team Size", f"{avg_team_size:.1f}")
    with col5:
        skill_coverage = len(set(skill for emp in employees for skill in emp.skills).intersection(
            set(skill for proj in projects for skill in proj.required_skills)))
        st.metric("📈 Skill Coverage", f"{skill_coverage}")
    
    # Advanced visualizations
    st.markdown("### 📈 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Skills Gap Analysis", "👥 Team Composition", "📊 Matching Insights"])
    
    with tab1:
        st.markdown("#### Skills Supply vs Demand")
        
        # Calculate supply and demand
        supply = {}
        demand = {}
        
        for emp in employees:
            for skill in emp.skills:
                supply[skill] = supply.get(skill, 0) + 1
        
        for proj in projects:
            for skill in proj.required_skills:
                demand[skill] = demand.get(skill, 0) + 1
        
        # Create comparison dataframe
        all_skills = set(supply.keys()) | set(demand.keys())
        comparison_data = []
        
        for skill in all_skills:
            supply_count = supply.get(skill, 0)
            demand_count = demand.get(skill, 0)
            gap = supply_count - demand_count
            comparison_data.append({
                'Skill': skill,
                'Supply': supply_count,
                'Demand': demand_count,
                'Gap': gap,
                'Status': 'Surplus' if gap > 0 else 'Deficit' if gap < 0 else 'Balanced'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Filter to top 20 skills by demand
        top_skills_df = comparison_df.nlargest(20, 'Demand')
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_skills_df['Skill'],
            y=top_skills_df['Supply'],
            name='Supply (Available)',
            marker_color='#4CAF50'
        ))
        
        fig.add_trace(go.Bar(
            x=top_skills_df['Skill'],
            y=top_skills_df['Demand'],
            name='Demand (Required)',
            marker_color='#F44336'
        ))
        
        fig.update_layout(
            title='Skills Supply vs Demand Analysis',
            xaxis_title='Skills',
            yaxis_title='Count',
            barmode='group',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show skills with biggest deficits
        deficit_skills = comparison_df[comparison_df['Gap'] < 0].nlargest(10, 'Demand')
        if not deficit_skills.empty:
            st.markdown("#### ⚠️ Critical Skills Deficits")
            for _, row in deficit_skills.iterrows():
                st.error(f"**{row['Skill']}**: Demand: {row['Demand']} | Supply: {row['Supply']} | Gap: {row['Gap']}")
        
        # Show skills with biggest surpluses
        surplus_skills = comparison_df[comparison_df['Gap'] > 0].nlargest(10, 'Supply')
        if not surplus_skills.empty:
            st.markdown("#### ✅ Skills Surpluses")
            for _, row in surplus_skills.iterrows():
                st.success(f"**{row['Skill']}**: Supply: {row['Supply']} | Demand: {row['Demand']} | Surplus: {row['Gap']}")

    with tab2:
        st.markdown("#### Team Composition Analysis")
        
        # Experience distribution by role
        role_exp_data = {}
        for emp in employees:
            if emp.current_role not in role_exp_data:
                role_exp_data[emp.current_role] = []
            role_exp_data[emp.current_role].append(emp.experience_years)
        
        # Create box plot
        fig = go.Figure()
        
        for role, exp_data in role_exp_data.items():
            fig.add_trace(go.Box(
                y=exp_data,
                name=role,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title='Experience Distribution by Role',
            yaxis_title='Years of Experience',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Location analysis
        location_data = {}
        for emp in employees:
            location_data[emp.location] = location_data.get(emp.location, 0) + 1
        
        fig = px.pie(values=list(location_data.values()), names=list(location_data.keys()),
                    title="Employee Distribution by Location")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Matching Insights & Predictions")
        
        # Run sample matching for analytics
        master_agent = MasterMatchingAgent()
        matches = master_agent.find_matches(employees, projects, min_score=0.5)
        
        if matches:
            # Match score distribution
            match_scores = [m.match_score for m in matches]
            fig = px.histogram(x=match_scores, nbins=20, 
                             title="Distribution of Match Scores",
                             labels={'x': 'Match Score', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Success rate by project type
            project_matches = {}
            for match in matches:
                project = next(p for p in projects if p.id == match.project_id)
                if project.client_type not in project_matches:
                    project_matches[project.client_type] = []
                project_matches[project.client_type].append(match.match_score)
            
            avg_scores = {pt: sum(scores)/len(scores) for pt, scores in project_matches.items()}
            fig = px.bar(x=list(avg_scores.keys()), y=list(avg_scores.values()),
                         title="Average Match Score by Project Type",
                         labels={'x': 'Project Type', 'y': 'Average Match Score'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 5 best matches
            st.markdown("#### 🏆 Top 5 Best Matches")
            top_matches = sorted(matches, key=lambda x: x.match_score, reverse=True)[:5]
            
            for i, match in enumerate(top_matches, 1):
                employee = next(e for e in employees if e.id == match.employee_id)
                project = next(p for p in projects if p.id == match.project_id)
                
                st.markdown(f"**#{i}: {employee.name} → {project.name}**")
                st.progress(match.match_score, text=f"Score: {match.match_score:.2f}")
                st.caption(f"Skills: {', '.join(employee.skills[:3])}...")
                st.markdown("---")

def render_agent_info():
    """Detailed information about AI agents"""
    st.title("🤖 AI Agents Information Hub")
    
    st.markdown("""
    ## Meet Your AI Team
    
    This system uses a team of specialized AI agents working together to find optimal matches. 
    Each agent has a specific role and expertise area.
    """)
    
    agents_info = [
        {
            "name": "🎯 Skill Matching Agent",
            "icon": "🎯",
            "description": "Analyzes technical skills and requirements using advanced NLP techniques",
            "capabilities": [
                "TF-IDF semantic similarity analysis",
                "Direct skill overlap calculation",
                "Skill clustering and categorization",
                "Emerging technology recognition"
            ],
            "weight": "35-45% of final score",
            "techniques": ["Cosine Similarity", "N-gram Analysis", "Skill Taxonomy Matching"]
        },
        {
            "name": "📅 Availability Agent",
            "icon": "📅",
            "description": "Evaluates timing, schedule compatibility, and project timing constraints",
            "capabilities": [
                "Calendar gap analysis",
                "Project timing optimization",
                "Resource allocation forecasting",
                "Burnout risk assessment"
            ],
            "weight": "20-35% of final score",
            "techniques": ["Time Series Analysis", "Gap Detection", "Workload Balancing"]
        },
        {
            "name": "👨‍💼 Experience Agent",
            "icon": "👨‍💼",
            "description": "Assesses experience levels, career progression, and role suitability",
            "capabilities": [
                "Experience level matching",
                "Career path alignment",
                "Growth potential assessment",
                "Over/under qualification detection"
            ],
            "weight": "15-25% of final score",
            "techniques": ["Experience Gradient Analysis", "Career Path Modeling", "Skill Progression Tracking"]
        },
        {
            "name": "🗺️ Location Agent",
            "icon": "🗺️",
            "description": "Handles geographic constraints, remote work compatibility, and travel requirements",
            "capabilities": [
                "Geographic proximity analysis",
                "Remote work compatibility",
                "Travel time optimization",
                "Time zone alignment"
            ],
            "weight": "5-20% of final score",
            "techniques": ["Geospatial Analysis", "Remote Work Scoring", "Travel Time Optimization"]
        },
        {
            "name": "⭐ Performance Agent",
            "icon": "⭐",
            "description": "Evaluates past performance, quality metrics, and reliability factors",
            "capabilities": [
                "Performance rating analysis",
                "Quality assurance scoring",
                "Reliability assessment",
                "Peer feedback integration"
            ],
            "weight": "Bonus/Penalty modifier",
            "techniques": ["Performance Trend Analysis", "Peer Rating Integration", "Quality Metrics"]
        },
        {
            "name": "🧠 Master Coordination Agent",
            "icon": "🧠",
            "description": "Orchestrates all specialized agents and makes final recommendations",
            "capabilities": [
                "Dynamic weight adjustment",
                "Conflict resolution",
                "Confidence level calculation",
                "Multi-criteria optimization"
            ],
            "weight": "Final decision maker",
            "techniques": ["Ensemble Learning", "Multi-objective Optimization", "Confidence Scoring"]
        }
    ]
    
    for agent in agents_info:
        with st.expander(f"{agent['icon']} {agent['name']}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {agent['description']}")
                st.markdown("**Key Capabilities:**")
                for capability in agent['capabilities']:
                    st.markdown(f"• {capability}")
            
            with col2:
                st.metric("Weight in Scoring", agent['weight'])
                st.markdown("**Techniques Used:**")
                for technique in agent['techniques']:
                    st.markdown(f"📊 {technique}")
    
    st.markdown("---")
    
    # Agent interaction demo
    st.markdown("### 🎭 Live Agent Interaction Demo")
    
    if st.button("🔄 Run Agent Simulation", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate agent workflow
        agents = ["Skill Matching", "Availability", "Experience", "Location", "Performance", "Master"]
        for i, agent in enumerate(agents):
            progress = (i + 1) / len(agents)
            progress_bar.progress(progress)
            status_text.text(f"🤖 {agent} Agent processing...")
            st.session_state[f"agent_{i}"] = True
            st.rerun()
            import time
            time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ All agents completed their analysis successfully!")
        
        # Show agent results
        st.markdown("#### 📊 Agent Contribution Breakdown")
        
        contributions = {
            "Skill Matching": 40,
            "Availability": 25,
            "Experience": 20,
            "Location": 10,
            "Performance": 5
        }
        
        fig = px.pie(values=list(contributions.values()), names=list(contributions.keys()),
                    title="Typical Agent Contribution to Final Score")
        st.plotly_chart(fig, use_container_width=True)

# Main application logic
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'employees' not in st.session_state:
        st.session_state.employees = DataManager.get_sample_employees()
    if 'projects' not in st.session_state:
        st.session_state.projects = DataManager.get_sample_projects()
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Render appropriate page based on selection
    if current_page == "🏠 Dashboard":
        render_dashboard()
    elif current_page == "🎯 AI Matching":
        render_matching_page()
    elif current_page == "👥 Employees":
        render_employee_management()
    elif current_page == "📋 Projects":
        render_project_management()
    elif current_page == "📊 Analytics":
        render_analytics()
    elif current_page == "🤖 AI Agents Info":
        render_agent_info()

if __name__ == "__main__":
    main()