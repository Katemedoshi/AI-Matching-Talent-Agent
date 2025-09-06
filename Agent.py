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
    """Enhanced sidebar with agent information and navigation"""
    st.sidebar.title("🚀 Navigation")
    st.sidebar.markdown("---")
    
    # Initialize current page in session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "🏠 Dashboard"
    
    # Get the current page selection
    page_options = ["🏠 Dashboard", "🎯 AI Matching", "👥 Employees", "📋 Projects", "📊 Analytics", "🤖 AI Agents Info"]
    
    # Find current page index
    try:
        current_index = page_options.index(st.session_state['current_page'])
    except ValueError:
        current_index = 0
        st.session_state['current_page'] = "🏠 Dashboard"
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        page_options,
        index=current_index
    )
    
    # Update session state if selection changed
    if page != st.session_state['current_page']:
        st.session_state['current_page'] = page
        st.rerun()
    
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
    
    return st.session_state['current_page']

def render_dashboard():
    """Enhanced dashboard with better visualizations and fixed navigation"""
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
    
    # Quick actions with session state navigation
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Run AI Matching", type="primary", use_container_width=True):
            st.session_state['current_page'] = "🎯 AI Matching"
            st.rerun()
    with col2:
        if st.button("👥 Add New Employee", use_container_width=True):
            st.session_state['current_page'] = "👥 Employees"
            st.rerun()
    with col3:
        if st.button("📋 Create New Project", use_container_width=True):
            st.session_state['current_page'] = "📋 Projects"
            st.rerun()
    
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
        
        status_text.text('🧠 Calculating matches with AI agents...')
        progress_bar.progress(60)
        
        # Find matches
        matches = master_agent.find_matches(filtered_employees, filtered_projects, min_score, max_results)
        
        status_text.text('📊 Preparing results visualization...')
        progress_bar.progress(80)
        
        # Display results
        st.markdown("### 📊 Matching Results")
        
        if not matches:
            st.warning("❌ No matches found with the current criteria. Try adjusting your filters.")
        else:
            # Create DataFrame for visualization
            match_data = []
            for match in matches:
                emp = next((e for e in employees if e.id == match.employee_id), None)
                proj = next((p for p in projects if p.id == match.project_id), None)
                
                if emp and proj:
                    match_data.append({
                        'Employee': emp.name,
                        'Project': proj.name,
                        'Score': match.match_score,
                        'Skill Match': match.skill_match,
                        'Availability Match': match.availability_match,
                        'Experience Match': match.experience_match,
                        'Location Match': match.location_match,
                        'Confidence': match.confidence_level,
                        'Reasoning': match.reasoning
                    })
            
            match_df = pd.DataFrame(match_data)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Total Matches", len(matches))
            with col2:
                avg_score = match_df['Score'].mean() if not match_df.empty else 0
                st.metric("📈 Average Match Score", f"{avg_score:.2f}")
            with col3:
                high_conf = len(match_df[match_df['Confidence'] == 'High']) if not match_df.empty else 0
                st.metric("✅ High Confidence", high_conf)
            
            st.markdown("---")
            
            # Interactive visualization
            st.markdown("#### 📈 Match Score Distribution")
            fig = px.histogram(match_df, x='Score', title="Distribution of Match Scores",
                              color='Confidence', color_discrete_map={'High': '#4CAF50', 'Medium': '#FF9800', 'Low': '#F44336'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("#### 📋 Detailed Match Results")
            
            for idx, row in match_df.iterrows():
                with st.expander(f"👤 {row['Employee']} → 🎯 {row['Project']} | Score: {row['Score']:.2f} | Confidence: {row['Confidence']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Component Scores:**")
                        st.markdown(f"- 🎯 Skill Match: `{row['Skill Match']:.2f}`")
                        st.markdown(f"- 📅 Availability: `{row['Availability Match']:.2f}`")
                        st.markdown(f"- 👨‍💼 Experience: `{row['Experience Match']:.2f}`")
                        st.markdown(f"- 🗺️ Location: `{row['Location Match']:.2f}`")
                    
                    with col2:
                        st.markdown("**🤖 AI Reasoning:**")
                        if show_reasoning:
                            st.info(row['Reasoning'])
                        else:
                            st.info("Enable 'Show AI Reasoning' to view detailed analysis")
            
            # Download option
            st.markdown("---")
            st.download_button(
                label="📥 Download Results as CSV",
                data=match_df.to_csv(index=False),
                file_name="ai_matching_results.csv",
                mime="text/csv"
            )
        
        progress_bar.progress(100)
        status_text.text('✅ Matching analysis completed!')

def render_employees_page():
    """Enhanced employee management page"""
    st.title("👥 Employee Management")
    
    # Load data
    employees = st.session_state.get('employees', DataManager.get_sample_employees())
    
    # Display employee list
    st.markdown("### 📋 Employee Directory")
    
    for emp in employees:
        with st.expander(f"👤 {emp.name} - {emp.current_role}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**📧 Email:** {emp.email}")
                st.markdown(f"**📍 Location:** {emp.location}")
                st.markdown(f"**📅 Availability:** {emp.availability}")
                st.markdown(f"**⭐ Performance Rating:** {emp.performance_rating}")
                
            with col2:
                st.markdown(f"**👨‍💼 Experience:** {emp.experience_years} years")
                st.markdown(f"**💰 Salary Range:** {emp.salary_range}")
                st.markdown(f"**📅 Last Project End:** {emp.last_project_end}")
                
            st.markdown("**🛠️ Skills:** " + ", ".join(emp.skills))
            st.markdown("**🎯 Career Interests:** " + ", ".join(emp.career_interests))
            if emp.certifications:
                st.markdown("**📜 Certifications:** " + ", ".join(emp.certifications))
    
    # Add new employee form
    st.markdown("---")
    st.markdown("### ➕ Add New Employee")
    
    with st.form("add_employee_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            location = st.text_input("Location")
            current_role = st.text_input("Current Role")
            experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
            
        with col2:
            availability = st.selectbox("Availability", ["Available", "Partial", "Busy"])
            performance = st.slider("Performance Rating", 1.0, 5.0, 4.0, 0.1)
            last_project_end = st.date_input("Last Project End Date", datetime.now().date())
            salary_range = st.text_input("Salary Range")
            
        skills = st.text_input("Skills (comma separated)")
        career_interests = st.text_input("Career Interests (comma separated)")
        certifications = st.text_input("Certifications (comma separated)")
        
        submitted = st.form_submit_button("Add Employee")
        
        if submitted:
            if name and email:
                new_employee = Employee(
                    id=f"emp_{len(employees)+1:03d}",
                    name=name,
                    email=email,
                    skills=[s.strip() for s in skills.split(",")] if skills else [],
                    experience_years=experience,
                    current_role=current_role,
                    availability=availability,
                    career_interests=[i.strip() for i in career_interests.split(",")] if career_interests else [],
                    location=location,
                    performance_rating=performance,
                    last_project_end=last_project_end.strftime("%Y-%m-%d"),
                    salary_range=salary_range,
                    certifications=[c.strip() for c in certifications.split(",")] if certifications else []
                )
                
                employees.append(new_employee)
                st.session_state.employees = employees
                st.success(f"✅ Employee {name} added successfully!")
                st.rerun()
            else:
                st.error("❌ Name and email are required fields")

def render_projects_page():
    """Enhanced project management page"""
    st.title("📋 Project Management")
    
    # Load data
    projects = st.session_state.get('projects', DataManager.get_sample_projects())
    
    # Display project list
    st.markdown("### 📋 Project Directory")
    
    for proj in projects:
        with st.expander(f"🎯 {proj.name} - {proj.urgency} Priority"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**📝 Description:** {proj.description}")
                st.markdown(f"**📍 Location:** {proj.location}")
                st.markdown(f"**⏱️ Duration:** {proj.duration_months} months")
                st.markdown(f"**👥 Team Size:** {proj.team_size} people")
                
            with col2:
                st.markdown(f"**🚨 Urgency:** {proj.urgency}")
                st.markdown(f"**👨‍💼 Experience Required:** {proj.experience_required} years")
                st.markdown(f"**💰 Budget Category:** {proj.budget_category}")
                st.markdown(f"**🏢 Client Type:** {proj.client_type}")
                st.markdown(f"**📅 Start Date:** {proj.start_date}")
                
            st.markdown("**🛠️ Required Skills:** " + ", ".join(proj.required_skills))
    
    # Add new project form
    st.markdown("---")
    st.markdown("### ➕ Add New Project")
    
    with st.form("add_project_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Project Name")
            description = st.text_area("Description")
            location = st.text_input("Location")
            duration = st.number_input("Duration (months)", min_value=1, max_value=36, value=6)
            team_size = st.number_input("Team Size", min_value=1, max_value=20, value=3)
            
        with col2:
            urgency = st.selectbox("Urgency", ["High", "Medium", "Low"])
            experience_required = st.number_input("Experience Required (years)", min_value=0, max_value=30, value=3)
            budget_category = st.selectbox("Budget Category", ["High", "Medium", "Low"])
            client_type = st.text_input("Client Type")
            start_date = st.date_input("Start Date", datetime.now().date() + timedelta(days=30))
            
        required_skills = st.text_input("Required Skills (comma separated)")
        
        submitted = st.form_submit_button("Add Project")
        
        if submitted:
            if name and description:
                new_project = Project(
                    id=f"proj_{len(projects)+1:03d}",
                    name=name,
                    description=description,
                    required_skills=[s.strip() for s in required_skills.split(",")] if required_skills else [],
                    duration_months=duration,
                    urgency=urgency,
                    location=location,
                    team_size=team_size,
                    experience_required=experience_required,
                    budget_category=budget_category,
                    client_type=client_type,
                    start_date=start_date.strftime("%Y-%m-%d")
                )
                
                projects.append(new_project)
                st.session_state.projects = projects
                st.success(f"✅ Project {name} added successfully!")
                st.rerun()
            else:
                st.error("❌ Project name and description are required fields")

def render_analytics_page():
    """Enhanced analytics page with visualizations"""
    st.title("📊 Analytics & Insights")
    
    # Load data
    employees = st.session_state.get('employees', DataManager.get_sample_employees())
    projects = st.session_state.get('projects', DataManager.get_sample_projects())
    
    # Skill analysis
    st.markdown("### 🛠️ Skill Analysis")
    
    # Extract all skills
    all_skills = []
    for emp in employees:
        all_skills.extend(emp.skills)
    
    skill_counts = pd.Series(all_skills).value_counts().head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(x=skill_counts.values, y=skill_counts.index, orientation='h',
                    title="Top 10 Skills in Workforce",
                    labels={'x': 'Count', 'y': 'Skill'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Skill gaps analysis
        required_skills = []
        for proj in projects:
            required_skills.extend(proj.required_skills)
        
        req_skill_counts = pd.Series(required_skills).value_counts().head(10)
        
        fig = px.bar(x=req_skill_counts.values, y=req_skill_counts.index, orientation='h',
                    title="Top 10 Required Skills in Projects",
                    labels={'x': 'Count', 'y': 'Skill'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Experience and performance correlation
    st.markdown("### 👥 Workforce Analytics")
    
    emp_data = []
    for emp in employees:
        emp_data.append({
            'Name': emp.name,
            'Experience': emp.experience_years,
            'Performance': emp.performance_rating,
            'Location': emp.location,
            'Availability': emp.availability
        })
    
    emp_df = pd.DataFrame(emp_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(emp_df, x='Experience', y='Performance', color='Availability',
                        title="Experience vs Performance by Availability",
                        hover_data=['Name', 'Location'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        location_counts = emp_df['Location'].value_counts()
        fig = px.pie(values=location_counts.values, names=location_counts.index,
                    title="Employee Distribution by Location")
        st.plotly_chart(fig, use_container_width=True)
    
    # Project analysis
    st.markdown("### 📋 Project Analytics")
    
    proj_data = []
    for proj in projects:
        proj_data.append({
            'Name': proj.name,
            'Duration': proj.duration_months,
            'Team Size': proj.team_size,
            'Urgency': proj.urgency,
            'Experience Required': proj.experience_required
        })
    
    proj_df = pd.DataFrame(proj_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(proj_df, x='Urgency', y='Duration', color='Urgency',
                    title="Project Duration by Urgency Level")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(proj_df, x='Team Size', y='Experience Required', color='Urgency',
                        title="Team Size vs Experience Required",
                        hover_data=['Name'])
        st.plotly_chart(fig, use_container_width=True)

def render_agents_info_page():
    """Page with information about the AI agents"""
    st.title("🤖 AI Agents Information")
    
    st.markdown("""
    ## Specialized AI Agents Powering the Matching System
    
    This platform uses a multi-agent architecture where specialized AI agents work together
    to find optimal matches between employees and projects.
    """)
    
    # Agent cards
    agents_info = [
        {
            "name": "🎯 Skill Matching Agent",
            "description": "Analyzes technical skills compatibility using TF-IDF and cosine similarity",
            "responsibilities": [
                "Processes skill keywords from both employees and projects",
                "Calculates semantic similarity between skill sets",
                "Identifies skill gaps and overlaps",
                "Provides detailed reasoning about skill matches"
            ]
        },
        {
            "name": "📅 Availability Agent",
            "description": "Evaluates timing and scheduling compatibility",
            "responsibilities": [
                "Assesses employee availability status",
                "Calculates optimal project start timing",
                "Identifies potential scheduling conflicts",
                "Considers time between projects for optimal resource allocation"
            ]
        },
        {
            "name": "👨‍💼 Experience Agent",
            "description": "Matches experience levels and career progression",
            "responsibilities": [
                "Compares employee experience with project requirements",
                "Identifies optimal experience matches (not too junior, not too senior)",
                "Considers career development opportunities",
                "Flags significant experience gaps or overqualification"
            ]
        },
        {
            "name": "🗺️ Location Agent",
            "description": "Handles geographic and remote work considerations",
            "responsibilities": [
                "Matches physical locations and remote work compatibility",
                "Identifies regional proximity for reduced travel needs",
                "Flags significant location mismatches",
                "Supports distributed team configurations"
            ]
        },
        {
            "name": "⭐ Performance Agent",
            "description": "Incorporates performance history into matching",
            "responsibilities": [
                "Applies performance-based bonuses or adjustments",
                "Prioritizes high performers for critical projects",
                "Supports development opportunities for improving employees",
                "Maintains quality standards through performance consideration"
            ]
        },
        {
            "name": "🧠 Master Matching Agent",
            "description": "Orchestrates all specialized agents for final recommendations",
            "responsibilities": [
                "Coordinates all specialized agent analyses",
                "Applies dynamic weighting based on project urgency",
                "Generates comprehensive match scores",
                "Provides final recommendations with confidence levels"
            ]
        }
    ]
    
    for agent in agents_info:
        with st.expander(agent["name"]):
            st.markdown(f"**Description:** {agent['description']}")
            st.markdown("**Key Responsibilities:**")
            for responsibility in agent["responsibilities"]:
                st.markdown(f"- {responsibility}")

# Main application logic
def main():
    """Main application function"""
    # Initialize session state
    if 'employees' not in st.session_state:
        st.session_state.employees = DataManager.get_sample_employees()
    
    if 'projects' not in st.session_state:
        st.session_state.projects = DataManager.get_sample_projects()
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Render the appropriate page based on selection
    if current_page == "🏠 Dashboard":
        render_dashboard()
    elif current_page == "🎯 AI Matching":
        render_matching_page()
    elif current_page == "👥 Employees":
        render_employees_page()
    elif current_page == "📋 Projects":
        render_projects_page()
    elif current_page == "📊 Analytics":
        render_analytics_page()
    elif current_page == "🤖 AI Agents Info":
        render_agents_info_page()

if __name__ == "__main__":
    main()
